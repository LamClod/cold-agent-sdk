use crate::error::AgentError;
use crate::result::AgentResult;

/// Trait for observing agent lifecycle events.
///
/// All methods have default no-op implementations so callers only need to
/// override the events they care about.
pub trait AgentCallback: Send + Sync {
    /// Called when the model emits text content.
    fn on_text(&self, _text: &str) {}
    /// Called when the model requests a tool invocation.
    fn on_tool_call(&self, _name: &str, _args: &serde_json::Value) {}
    /// Called after a tool finishes execution.
    fn on_tool_result(&self, _name: &str, _result: &cold_tools::ToolResult) {}
    /// Called when a non-fatal error occurs during the loop.
    fn on_error(&self, _error: &AgentError) {}
    /// Called when the agent loop finishes successfully.
    fn on_complete(&self, _result: &AgentResult) {}
    /// Called after a successful context compression.
    fn on_compress(&self, _before_tokens: u32, _after_tokens: u32) {}
    /// Called at the start of each iteration.
    fn on_progress(&self, _turn: u32, _max_turns: u32) {}
}

/// No-op callback that discards every event.
pub struct SilentCallback;

impl AgentCallback for SilentCallback {}

/// Callback that prints events to stdout.
pub struct PrintCallback;

impl AgentCallback for PrintCallback {
    fn on_text(&self, text: &str) {
        print!("{text}");
    }

    fn on_tool_call(&self, name: &str, _args: &serde_json::Value) {
        println!("[tool] {name}(...)");
    }

    fn on_tool_result(&self, name: &str, result: &cold_tools::ToolResult) {
        let preview = result.as_text();
        let preview = if preview.len() > 120 {
            &preview[..120]
        } else {
            preview
        };
        println!("[result] {name}: {preview}");
    }
}
