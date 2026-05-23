/// The final outcome of an [`Agent::run`](crate::Agent::run) invocation.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// The assistant's final text reply.
    pub text: String,
    /// Number of agentic turns consumed.
    pub turns_used: u32,
    /// Aggregated token usage across all API calls.
    pub tokens: TokenUsage,
    /// Record of every tool call made during the run.
    pub tools_called: Vec<ToolCallRecord>,
    /// Whether context compression was triggered.
    pub compressed: bool,
}

/// Aggregated token usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    /// Total prompt tokens sent.
    pub prompt_tokens: u32,
    /// Total completion tokens received.
    pub completion_tokens: u32,
    /// Sum of prompt + completion tokens.
    pub total_tokens: u32,
}

/// A record of a single tool invocation.
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// Tool name.
    pub name: String,
    /// Arguments passed to the tool.
    pub args: serde_json::Value,
    /// First 200 characters of the tool output.
    pub result_preview: String,
    /// Wall-clock execution time in milliseconds.
    pub duration_ms: u64,
    /// Whether the tool returned a successful result.
    pub succeeded: bool,
}
