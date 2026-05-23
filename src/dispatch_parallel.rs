use std::sync::Arc;

use cold_sdk::ToolCall;
use cold_tools::{Dispatcher, ToolContext};

use crate::callback::AgentCallback;
use crate::streaming_executor::ExecutionResult;

/// Execute tool calls, routing through [`Dispatcher::execute_batch`] for true
/// parallel execution of concurrency-safe tools.
///
/// Callbacks (`on_tool_call` / `on_tool_result`) fire before and after the
/// batch so every tool still gets a notification.
pub async fn dispatch_with_parallelism(
    tool_calls: &[ToolCall],
    dispatcher: &mut Dispatcher,
    ctx: &ToolContext,
    callback: &Arc<dyn AgentCallback>,
) -> Vec<ExecutionResult> {
    if tool_calls.is_empty() {
        return Vec::new();
    }

    // Parse arguments and fire pre-call callbacks for every tool.
    let mut parsed_args: Vec<serde_json::Value> = Vec::with_capacity(tool_calls.len());
    for tc in tool_calls {
        let args: serde_json::Value =
            serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        callback.on_tool_call(&tc.function.name, &args);
        parsed_args.push(args);
    }

    // Build the batch input expected by Dispatcher::execute_batch.
    let calls: Vec<(String, serde_json::Value)> = tool_calls
        .iter()
        .zip(parsed_args.iter())
        .map(|(tc, args)| (tc.function.name.clone(), args.clone()))
        .collect();

    let start = std::time::Instant::now();
    let batch_results = dispatcher.execute_batch(calls, ctx).await;
    #[allow(clippy::cast_possible_truncation)]
    let batch_duration_ms = start.elapsed().as_millis() as u64;

    // Map batch results back to ExecutionResults and fire post-call callbacks.
    let per_tool_ms = if tool_calls.is_empty() {
        0
    } else {
        batch_duration_ms / tool_calls.len() as u64
    };

    tool_calls
        .iter()
        .zip(batch_results)
        .map(|(tc, result)| {
            if let Ok(ref r) = result {
                callback.on_tool_result(&tc.function.name, r);
            }
            ExecutionResult {
                tool_call_id: tc.id.clone(),
                tool_name: tc.function.name.clone(),
                result,
                duration_ms: per_tool_ms,
            }
        })
        .collect()
}
