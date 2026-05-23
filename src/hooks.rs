//! Hook system for observing and intercepting agent lifecycle events.
//!
//! Unlike [`AgentCallback`](crate::callback::AgentCallback) which is purely
//! observational, hooks can influence control flow by returning
//! [`HookDecision::Block`].

/// Decision returned by hook methods that support interception.
#[derive(Debug, Clone)]
pub enum HookDecision {
    /// Allow the operation to proceed.
    Continue,
    /// Block the operation with the given reason.
    Block(String),
}

/// Trait for intercepting agent lifecycle events.
///
/// All methods have default no-op implementations. Implement only the
/// events you care about.
pub trait AgentHook: Send + Sync {
    /// Whether this hook is a real (non-noop) hook. Returns `true` by default.
    /// Only [`NoopHook`] overrides this to `false`. Used to decide whether to
    /// include hook guidance in the system prompt.
    fn is_active(&self) -> bool {
        true
    }

    /// Called once when a session starts (before the first API call).
    fn on_session_start(&self, _session_id: &str) {}

    /// Called before each tool execution. Return `Block` to skip the tool.
    fn on_pre_tool_call(&self, _name: &str, _args: &serde_json::Value) -> HookDecision {
        HookDecision::Continue
    }

    /// Called after each tool execution completes.
    fn on_post_tool_call(&self, _name: &str, _result: &cold_tools::ToolResult) {}

    /// Called before context compression begins.
    fn on_pre_compact(&self, _message_count: usize) {}

    /// Called after context compression completes.
    fn on_post_compact(&self, _before_tokens: u32, _after_tokens: u32) {}

    /// Called at the end of each agentic turn.
    fn on_turn_complete(&self, _turn: u32) {}

    /// Called when the model returns a stop finish reason.
    /// Return `Block` to force the loop to continue instead of stopping.
    fn on_stop(&self) -> HookDecision {
        HookDecision::Continue
    }
}

/// No-op hook that allows everything and observes nothing.
pub struct NoopHook;

impl AgentHook for NoopHook {
    fn is_active(&self) -> bool {
        false
    }
}
