use std::fmt;

/// Errors that can occur during agent orchestration.
#[derive(Debug)]
pub enum AgentError {
    /// Error from the underlying SDK (API calls).
    Sdk(cold_sdk::ColdError),
    /// Error from the context compressor.
    Context(cold_context::ContextError),
    /// Error from tool execution.
    Tool(cold_tools::ToolError),
    /// Configuration error.
    Config(String),
    /// Session I/O error (save/load).
    SessionIo(std::io::Error),
    /// The iteration budget has been exhausted.
    BudgetExhausted {
        /// Number of turns consumed.
        turns_used: u32,
        /// Maximum turns allowed.
        max_turns: u32,
    },
    /// The agent loop was interrupted.
    Interrupted,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sdk(e) => write!(f, "sdk error: {e}"),
            Self::Context(e) => write!(f, "context error: {e}"),
            Self::Tool(e) => write!(f, "tool error: {e}"),
            Self::Config(msg) => write!(f, "config error: {msg}"),
            Self::SessionIo(e) => write!(f, "session I/O error: {e}"),
            Self::BudgetExhausted {
                turns_used,
                max_turns,
            } => write!(
                f,
                "budget exhausted: used {turns_used} of {max_turns} turns"
            ),
            Self::Interrupted => write!(f, "agent interrupted"),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sdk(e) => Some(e),
            Self::Context(e) => Some(e),
            Self::Tool(e) => Some(e),
            Self::SessionIo(e) => Some(e),
            _ => None,
        }
    }
}

impl From<cold_sdk::ColdError> for AgentError {
    fn from(e: cold_sdk::ColdError) -> Self {
        Self::Sdk(e)
    }
}

impl From<cold_context::ContextError> for AgentError {
    fn from(e: cold_context::ContextError) -> Self {
        Self::Context(e)
    }
}

impl From<cold_tools::ToolError> for AgentError {
    fn from(e: cold_tools::ToolError) -> Self {
        Self::Tool(e)
    }
}
