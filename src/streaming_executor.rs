use cold_sdk::{FinishReason, StreamToolCall, ToolCall};
use cold_tools::{ToolContext, ToolError, ToolResult};
use std::sync::Arc;

use crate::callback::AgentCallback;
use crate::dispatch_parallel::dispatch_with_parallelism;

/// Accumulates streaming tool call fragments and executes tools as they complete.
///
/// Upgrade: eagerly validates JSON arguments during streaming so malformed
/// arguments are detected immediately, and tool execution can start the
/// instant the stream ends with `finish_reason=tool_calls`.
pub struct StreamingToolExecutor {
    /// Accumulated tool calls (index -> partial state).
    pending: Vec<ToolCallAccumulator>,
    /// Assembled text content from the stream.
    text: String,
    /// Finish reason from the stream.
    finish_reason: Option<FinishReason>,
    /// Tracks which tool call indices have already been validated as valid JSON.
    validated: Vec<bool>,
    /// Indices that failed eager JSON validation.
    invalid_args: Vec<usize>,
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// A tool call that has been fully assembled and eagerly validated.
#[derive(Debug)]
pub struct ReadyToolCall {
    /// The original index in the pending list.
    pub index: usize,
    /// The fully assembled tool call.
    pub tool_call: ToolCall,
}

/// Result of executing a single tool call.
pub struct ExecutionResult {
    /// The tool call ID returned by the model.
    pub tool_call_id: String,
    /// The tool name.
    pub tool_name: String,
    /// The execution result or error.
    pub result: Result<ToolResult, ToolError>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Maximum tool call index we accept (sanity bound).
const MAX_TOOL_CALL_INDEX: usize = 128;

impl StreamingToolExecutor {
    /// Create a new empty executor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
            text: String::new(),
            finish_reason: None,
            validated: Vec::new(),
            invalid_args: Vec::new(),
        }
    }

    /// Feed streaming tool call deltas from a single chunk.
    pub fn feed_tool_calls(&mut self, deltas: &[StreamToolCall]) {
        for delta in deltas {
            let idx = delta.index as usize;
            if idx > MAX_TOOL_CALL_INDEX {
                continue;
            }
            while self.pending.len() <= idx {
                self.pending.push(ToolCallAccumulator::default());
                self.validated.push(false);
            }
            let acc = &mut self.pending[idx];
            if let Some(ref id) = delta.id {
                if !id.is_empty() {
                    acc.id.clone_from(id);
                }
            }
            if let Some(ref f) = delta.function {
                if let Some(ref name) = f.name {
                    if !name.is_empty() {
                        acc.name.clone_from(name);
                    }
                }
                if let Some(ref args) = f.arguments {
                    acc.arguments.push_str(args);
                }
            }
        }
    }

    /// Feed streaming deltas and check which tool calls are fully assembled.
    ///
    /// A tool call is "ready" when it has an id, a name, and its accumulated
    /// arguments parse as valid JSON. This enables eager validation during
    /// streaming so errors surface early.
    pub fn feed_and_check(&mut self, deltas: &[StreamToolCall]) -> Vec<ReadyToolCall> {
        self.feed_tool_calls(deltas);

        let mut ready = Vec::new();
        for (idx, acc) in self.pending.iter().enumerate() {
            if idx < self.validated.len() && self.validated[idx] {
                continue; // already validated
            }
            if acc.id.is_empty() || acc.name.is_empty() || acc.arguments.is_empty() {
                continue;
            }
            if serde_json::from_str::<serde_json::Value>(&acc.arguments).is_ok() {
                if idx < self.validated.len() {
                    self.validated[idx] = true;
                }
                ready.push(ReadyToolCall {
                    index: idx,
                    tool_call: ToolCall {
                        id: acc.id.clone(),
                        call_type: "function".to_string(),
                        function: cold_sdk::FunctionCall {
                            name: acc.name.clone(),
                            arguments: acc.arguments.clone(),
                        },
                    },
                });
            }
        }
        ready
    }

    /// Feed text content from a delta.
    pub fn feed_text(&mut self, content: &str) {
        self.text.push_str(content);
    }

    /// Record the finish reason.
    pub const fn set_finish_reason(&mut self, reason: FinishReason) {
        self.finish_reason = Some(reason);
    }

    /// Get the accumulated text content.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the finish reason, if the stream has ended.
    #[must_use]
    pub const fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }

    /// Whether tool calls were received.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Validate all pending tool call arguments eagerly. Call this at stream
    /// end to detect malformed JSON before attempting execution.
    ///
    /// Returns indices of tool calls with invalid JSON arguments.
    pub fn validate_all_args(&mut self) -> Vec<usize> {
        self.invalid_args.clear();
        for (idx, acc) in self.pending.iter().enumerate() {
            if idx < self.validated.len() && self.validated[idx] {
                continue;
            }
            if serde_json::from_str::<serde_json::Value>(&acc.arguments).is_err() {
                self.invalid_args.push(idx);
            }
        }
        self.invalid_args.clone()
    }

    /// Take all assembled tool calls, consuming the pending state.
    pub fn take_tool_calls(&mut self) -> Vec<ToolCall> {
        self.validated.clear();
        self.invalid_args.clear();
        std::mem::take(&mut self.pending)
            .into_iter()
            .filter(|acc| !acc.id.is_empty() && !acc.name.is_empty())
            .map(|acc| ToolCall {
                id: acc.id,
                call_type: "function".to_string(),
                function: cold_sdk::FunctionCall {
                    name: acc.name,
                    arguments: acc.arguments,
                },
            })
            .collect()
    }

    /// Take the accumulated text, leaving an empty string in place.
    pub fn take_text(&mut self) -> String {
        std::mem::take(&mut self.text)
    }

    /// Execute all assembled tool calls against the given dispatcher.
    ///
    /// Concurrency-safe tools are run in parallel; others run serially.
    pub async fn execute_all(
        tool_calls: &[ToolCall],
        dispatcher: &mut cold_tools::Dispatcher,
        ctx: &ToolContext,
        callback: &Arc<dyn AgentCallback>,
    ) -> Vec<ExecutionResult> {
        dispatch_with_parallelism(tool_calls, dispatcher, ctx, callback).await
    }
}

impl Default for StreamingToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}
