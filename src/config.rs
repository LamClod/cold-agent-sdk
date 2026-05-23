use std::path::PathBuf;

use cold_context::CompressorConfig;
use cold_tools::PermissionMode;

/// Configuration for an [`Agent`](crate::Agent) instance.
pub struct AgentConfig {
    /// Model identifier (e.g. `"gpt-4o"`, `"claude-sonnet-4-20250514"`).
    pub model: String,
    /// Total context window size in tokens.
    pub context_length: u32,
    /// API key for the LLM provider.
    pub api_key: String,
    /// Custom API base URL. When `None`, the SDK default is used.
    pub base_url: Option<String>,
    /// Maximum agentic loop iterations before giving up.
    pub max_turns: u32,
    /// Project root directory for tool sandboxing.
    pub root_dir: PathBuf,
    /// Custom system prompt. Falls back to the built-in identity prompt.
    pub system_prompt: Option<String>,
    /// Directory containing skill files (`SKILL.md`).
    pub skills_dir: Option<PathBuf>,
    /// Whether to use streaming for API calls (reserved for future use).
    pub streaming: bool,
    /// Automatically compress context when the threshold is reached.
    pub auto_compress: bool,
    /// Persist the session to disk after each `run()`.
    pub persist_session: bool,
    /// Directory for session JSON files.
    pub session_dir: Option<PathBuf>,
    /// Override for compressor configuration.
    pub compressor_config: Option<CompressorConfig>,
    /// Permission mode controlling tool execution policy.
    pub permission_mode: PermissionMode,
    /// Fallback model identifier used when the primary model returns a
    /// non-retryable API error (4xx except 429).
    pub fallback_model: Option<String>,
    /// Custom API base URL for the fallback model.
    pub fallback_base_url: Option<String>,
    /// HTTP proxy URL (e.g. `"http://127.0.0.1:7890"`).
    pub proxy: Option<String>,
}

impl AgentConfig {
    /// Create a new config with sensible defaults.
    ///
    /// Only the three required fields are taken as arguments; everything else
    /// can be customised through the builder methods.
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        context_length: u32,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            context_length,
            api_key: api_key.into(),
            base_url: None,
            max_turns: 90,
            root_dir: PathBuf::from("."),
            system_prompt: None,
            skills_dir: None,
            streaming: true,
            auto_compress: true,
            persist_session: true,
            session_dir: None,
            compressor_config: None,
            permission_mode: PermissionMode::Default,
            fallback_model: None,
            fallback_base_url: None,
            proxy: None,
        }
    }

    /// Set a custom API base URL (without `/v1`).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set an HTTP proxy (e.g. `"http://127.0.0.1:7890"`).
    #[must_use]
    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self
    }

    /// Set the project root directory.
    #[must_use]
    pub fn with_root_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.root_dir = dir.into();
        self
    }

    /// Override the default system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum number of agentic turns.
    #[must_use]
    pub const fn with_max_turns(mut self, max: u32) -> Self {
        self.max_turns = max;
        self
    }

    /// Enable or disable streaming.
    #[must_use]
    pub const fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Set the session persistence directory.
    #[must_use]
    pub fn with_session_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.session_dir = Some(dir.into());
        self
    }

    /// Set the skills directory.
    #[must_use]
    pub fn with_skills_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.skills_dir = Some(dir.into());
        self
    }

    /// Enable or disable automatic context compression.
    #[must_use]
    pub const fn with_auto_compress(mut self, enabled: bool) -> Self {
        self.auto_compress = enabled;
        self
    }

    /// Enable or disable session persistence.
    #[must_use]
    pub const fn with_persist_session(mut self, enabled: bool) -> Self {
        self.persist_session = enabled;
        self
    }

    /// Set the permission mode for tool execution policy.
    #[must_use]
    pub const fn with_permission_mode(mut self, mode: PermissionMode) -> Self {
        self.permission_mode = mode;
        self
    }

    /// Set a fallback model for automatic failover on non-retryable errors.
    #[must_use]
    pub fn with_fallback_model(mut self, model: impl Into<String>) -> Self {
        self.fallback_model = Some(model.into());
        self
    }

    /// Set the API base URL for the fallback model.
    #[must_use]
    pub fn with_fallback_base_url(mut self, url: impl Into<String>) -> Self {
        self.fallback_base_url = Some(url.into());
        self
    }
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("model", &self.model)
            .field("context_length", &self.context_length)
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("max_turns", &self.max_turns)
            .field("root_dir", &self.root_dir)
            .field("system_prompt", &self.system_prompt.as_deref().map(|s| {
                if s.len() > 60 {
                    format!("{}...", &s[..60])
                } else {
                    s.to_string()
                }
            }))
            .field("skills_dir", &self.skills_dir)
            .field("streaming", &self.streaming)
            .field("auto_compress", &self.auto_compress)
            .field("persist_session", &self.persist_session)
            .field("session_dir", &self.session_dir)
            .field("compressor_config", &self.compressor_config)
            .field("permission_mode", &self.permission_mode)
            .field("fallback_model", &self.fallback_model)
            .field("fallback_base_url", &self.fallback_base_url)
            .finish()
    }
}
