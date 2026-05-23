use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cold_context::CompressorConfig;
use cold_sdk::{
    ChatMessage, ChatRequest, ColdClient, FinishReason, Tool as SdkTool,
};
use cold_tools::{
    AutoApprove, CoreToolConfig, Dispatcher, PermissionMode, ToolContext, ToolRegistry,
};

use crate::budget::IterationBudget;
use crate::callback::{AgentCallback, SilentCallback};
use crate::config::AgentConfig;
use crate::delegate::DelegateTool;
use crate::error::AgentError;
use crate::hooks::{AgentHook, HookDecision, NoopHook};
use crate::prompt::{build_system_prompt, strip_cache_boundary};
use crate::result::{AgentResult, TokenUsage, ToolCallRecord};
use crate::session::{append_to_log, save_session};
use crate::skill::SkillRegistry;
use crate::state::ConversationState;
use crate::streaming_executor::{ExecutionResult, StreamingToolExecutor};

/// The main agent orchestrator.
///
/// Glues together `cold-sdk` (API calls), `cold-context` (compression), and
/// `cold-tools` (tool execution) into a single agentic loop.
pub struct Agent {
    config: AgentConfig,
    client: ColdClient,
    /// Fallback client created on first non-retryable error, if configured.
    fallback_client: Option<ColdClient>,
    /// Whether we have switched to the fallback model for this session.
    using_fallback: bool,
    compressor: cold_context::ContextCompressor<cold_context::summary::LlmSummarizer>,
    tools: Arc<ToolRegistry>,
    dispatcher: Dispatcher,
    skills: SkillRegistry,
    state: ConversationState,
    budget: IterationBudget,
    callback: Arc<dyn AgentCallback>,
    hook: Arc<dyn AgentHook>,
    cancelled: Arc<AtomicBool>,
    /// Track which deferred tool names have been announced in this session.
    deferred_tools_announced: Vec<String>,
    /// Track which skill names are active in this session.
    active_skills: Vec<String>,
    /// Whether MCP tools are registered (for system prompt guidance).
    has_mcp_tools: bool,
}

impl Agent {
    /// Build a new agent from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Sdk` if the API client cannot be constructed, or
    /// `AgentError::SessionIo` if the skills directory cannot be read.
    pub fn new(config: AgentConfig) -> Result<Self, AgentError> {
        // 1. API client (with optional proxy)
        let mut client_config = if let Some(ref base_url) = config.base_url {
            cold_sdk::ClientConfig::with_endpoint(base_url, &config.api_key)
        } else {
            cold_sdk::ClientConfig::new(&config.api_key)
        };
        if let Some(ref proxy) = config.proxy {
            client_config = client_config.with_proxy(proxy);
        }
        let client = ColdClient::from_config(client_config)?;

        // 2. Compressor
        let compressor_config = config.compressor_config.clone().unwrap_or_else(|| {
            CompressorConfig::new(&config.model, config.context_length)
        });
        let compressor =
            cold_context::ContextCompressor::new(compressor_config, client.clone());

        // 3. Tool registry + core tools + delegate tool
        let mut tools = ToolRegistry::new();
        cold_tools::register_core_tools(
            &mut tools,
            CoreToolConfig {
                root_dir: config.root_dir.clone(),
                ..CoreToolConfig::default()
            },
        );
        tools.register(DelegateTool::new(client.clone(), &config));
        let tools = Arc::new(tools);

        // 4. Dispatcher
        let dispatcher = Dispatcher::new(Arc::clone(&tools));

        // 5. Skills
        let skills = if let Some(ref dir) = config.skills_dir {
            SkillRegistry::load_from_dir(dir)?
        } else {
            SkillRegistry::new()
        };

        // 6. State + budget
        let state = ConversationState::new();
        let budget = IterationBudget::new(config.max_turns);

        Ok(Self {
            config,
            client,
            fallback_client: None,
            using_fallback: false,
            compressor,
            tools,
            dispatcher,
            skills,
            state,
            budget,
            callback: Arc::new(SilentCallback),
            hook: Arc::new(NoopHook),
            cancelled: Arc::new(AtomicBool::new(false)),
            deferred_tools_announced: Vec::new(),
            active_skills: Vec::new(),
            has_mcp_tools: false,
        })
    }

    /// Attach a callback for lifecycle events.
    #[must_use]
    pub fn with_callback(mut self, cb: impl AgentCallback + 'static) -> Self {
        self.callback = Arc::new(cb);
        self
    }

    /// Attach a hook for intercepting lifecycle events.
    #[must_use]
    pub fn with_hook(mut self, hook: impl AgentHook + 'static) -> Self {
        self.hook = Arc::new(hook);
        self
    }

    /// Record that a deferred tool name was announced (for post-compact restoration).
    pub fn track_deferred_tool(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.deferred_tools_announced.contains(&name) {
            self.deferred_tools_announced.push(name);
        }
    }

    /// Record that a skill is active in this session (for post-compact restoration).
    pub fn track_active_skill(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.active_skills.contains(&name) {
            self.active_skills.push(name);
        }
    }

    /// Set whether MCP tools are registered (affects system prompt guidance).
    pub const fn set_has_mcp_tools(&mut self, has_mcp: bool) {
        self.has_mcp_tools = has_mcp;
    }

    /// Run the agentic loop with the given user prompt.
    ///
    /// This builds the system prompt, appends the user message, and enters the
    /// tool-use loop until the model stops, the budget is exhausted, or an
    /// unrecoverable error occurs.
    ///
    /// # Errors
    ///
    /// See [`AgentError`] for the full list of failure modes.
    pub async fn run(&mut self, prompt: &str) -> Result<AgentResult, AgentError> {
        self.state.last_user_message = prompt.to_string();
        self.hook.on_session_start(&self.state.session_id);

        // Determine hook presence (not NoopHook)
        let has_hooks = self.hook.is_active();

        // Build system prompt and seed the conversation
        let system = build_system_prompt(
            &self.config,
            &self.skills,
            &self.state,
            has_hooks,
            self.has_mcp_tools,
        );
        // Strip the cache boundary marker — it is a client-side signal and
        // should not appear in the text the model receives.
        let system = strip_cache_boundary(&system);
        self.state.add_message(ChatMessage::system(&system));
        self.state.add_message(ChatMessage::user(prompt));

        // JSONL append logging for the initial messages
        self.append_log_if_configured(self.state.messages.last().cloned())
            .await;

        self.run_loop().await
    }

    /// Continue the conversation without rebuilding the system prompt.
    ///
    /// Use this after `run()` to send follow-up messages within the same
    /// session.
    ///
    /// # Errors
    ///
    /// See [`AgentError`] for the full list of failure modes.
    pub async fn continue_with(&mut self, message: &str) -> Result<AgentResult, AgentError> {
        self.state.last_user_message = message.to_string();
        let msg = ChatMessage::user(message);
        self.state.add_message(msg.clone());
        self.append_log_if_configured(Some(msg)).await;
        self.run_loop().await
    }

    /// Clear all state and start fresh.
    pub fn reset(&mut self) {
        self.state = ConversationState::new();
        self.budget.reset();
        self.compressor.reset();
    }

    /// Persist the current session to disk.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Config` if no session directory is configured, or
    /// `AgentError::SessionIo` on filesystem errors.
    pub async fn save(&self) -> Result<(), AgentError> {
        let dir = self
            .config
            .session_dir
            .as_deref()
            .ok_or_else(|| AgentError::Config("no session_dir configured".into()))?;
        save_session(dir, &self.state, &self.config.model, &self.compressor).await
    }

    /// Restore a previously saved session.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Config` if message deserialization fails.
    pub fn restore_session(
        &mut self,
        session: crate::session::SavedSession,
    ) -> Result<(), AgentError> {
        self.state.session_id = session.session_id;
        self.state.turn_count = session.turn_count;
        self.state.messages = session
            .messages
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();
        self.compressor.restore_state(session.compressor_state);
        Ok(())
    }

    /// Signal the agent to stop after the current iteration.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    // ─── Internal ────────────────────────────────────────────────

    /// Whether an API error indicates a context-length-exceeded condition.
    fn is_context_length_error(err: &cold_sdk::ColdError) -> bool {
        matches!(
            err,
            cold_sdk::ColdError::Api { status: 400, body }
            if body.contains("context_length_exceeded")
                || body.contains("prompt_too_long")
        )
    }

    /// Attempt reactive compact recovery when the API returns a context length error.
    ///
    /// Clones the current messages, runs `reactive_compact` to drop oldest
    /// API-round groups, and restores the compacted messages if successful.
    /// Fires compress callback and hook events on success.
    ///
    /// Returns `Ok(true)` if recovery succeeded and the caller should retry.
    #[allow(clippy::unnecessary_wraps)] // Result is intentional for future error paths
    fn try_reactive_recovery(&mut self) -> Result<bool, AgentError> {
        let compressor_config = self.config.compressor_config.clone().unwrap_or_else(|| {
            cold_context::CompressorConfig::new(&self.config.model, self.config.context_length)
        });
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_tokens = (f64::from(compressor_config.threshold_tokens()) * 0.8) as u32;

        let msgs = self.state.messages.clone();
        let original_len = msgs.len();

        self.hook.on_pre_compact(original_len);

        match cold_context::reactive_compact(msgs, target_tokens, 3) {
            Ok(result) => {
                #[allow(clippy::cast_possible_truncation)]
                let groups_tokens = (result.groups_dropped as u32).saturating_mul(500);
                let original_tokens = result.estimated_tokens + groups_tokens;
                self.callback.on_compress(original_tokens, result.estimated_tokens);
                self.hook.on_post_compact(original_tokens, result.estimated_tokens);
                self.state.messages = result.messages;

                // Post-compact restoration: re-announce deferred tools and active skills
                self.post_compact_restore();

                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    /// After a successful compression, re-inject context that the agent layer
    /// tracks but `cold-context` does not know about: deferred tool
    /// announcements and active skill summaries.
    fn post_compact_restore(&mut self) {
        // 1. Re-announce deferred tools (if any)
        if !self.deferred_tools_announced.is_empty() {
            let announcement = format!(
                "[Available tools refreshed after compaction]\n\
                 The following tools are available via tool_search: {}",
                self.deferred_tools_announced.join(", ")
            );
            self.state.add_message(ChatMessage::system(&announcement));
        }

        // 2. Re-inject active skill summaries
        if !self.active_skills.is_empty() {
            let skills_note = format!(
                "[Active skills refreshed after compaction]: {}",
                self.active_skills.join(", ")
            );
            self.state.add_message(ChatMessage::system(&skills_note));
        }
    }

    /// Append a message to the JSONL log if session persistence is configured.
    async fn append_log_if_configured(&self, msg: Option<ChatMessage>) {
        let Some(dir) = &self.config.session_dir else {
            return;
        };
        if let Some(msg) = msg {
            let _ = append_to_log(dir, &self.state.session_id, msg).await;
        }
    }

    /// Select the active client: fallback if switched, otherwise primary.
    fn active_client(&self) -> &ColdClient {
        if self.using_fallback {
            self.fallback_client.as_ref().unwrap_or(&self.client)
        } else {
            &self.client
        }
    }

    /// Try the primary client; on non-retryable 4xx (not 429), attempt
    /// fallback if configured. Returns `None` if fallback succeeded (caller
    /// should retry the request).
    async fn try_with_fallback(
        &mut self,
        request: &ChatRequest,
    ) -> Result<cold_sdk::ChatResponse, AgentError> {
        let result = self.active_client().chat(request).await;
        match result {
            Ok(resp) => Ok(resp),
            Err(ref e) if !self.using_fallback && self.should_try_fallback(e) => {
                self.callback.on_error(&AgentError::Sdk(cold_sdk::ColdError::Config(
                    format!("primary model failed: {e}, trying fallback"),
                )));
                self.init_fallback_client()?;
                // Retry with fallback.
                self.active_client()
                    .chat(request)
                    .await
                    .map_err(AgentError::Sdk)
            }
            Err(e) => Err(AgentError::Sdk(e)),
        }
    }

    /// Whether a fallback attempt is warranted for this error.
    const fn should_try_fallback(&self, err: &cold_sdk::ColdError) -> bool {
        if self.config.fallback_model.is_none() {
            return false;
        }
        matches!(err, cold_sdk::ColdError::Api { status, .. } if *status >= 400 && *status != 429 && *status < 500)
    }

    /// Construct the fallback client and switch to it.
    fn init_fallback_client(&mut self) -> Result<(), AgentError> {
        let fallback_model = self
            .config
            .fallback_model
            .as_ref()
            .ok_or_else(|| AgentError::Config("no fallback model configured".into()))?;

        let fb_base_url = self
            .config
            .fallback_base_url
            .as_deref()
            .or(self.config.base_url.as_deref());

        let mut fb_cfg = if let Some(url) = fb_base_url {
            cold_sdk::ClientConfig::with_endpoint(url, &self.config.api_key)
        } else {
            cold_sdk::ClientConfig::new(&self.config.api_key)
        };
        if let Some(ref proxy) = self.config.proxy {
            fb_cfg = fb_cfg.with_proxy(proxy);
        }
        let fb_client = ColdClient::from_config(fb_cfg)?;

        self.fallback_client = Some(fb_client);
        self.using_fallback = true;
        // Update the model in the config so subsequent requests use it.
        self.config.model.clone_from(fallback_model);
        Ok(())
    }

    /// The core agentic loop shared by `run` and `continue_with`.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    async fn run_loop(&mut self) -> Result<AgentResult, AgentError> {
        let mut result_text = String::new();
        let mut tools_called: Vec<ToolCallRecord> = Vec::new();
        let mut compressed = false;
        let mut output_recovery_count: u32 = 0;

        while self.budget.has_remaining() {
            if self.cancelled.load(Ordering::Relaxed) {
                return Err(AgentError::Interrupted);
            }

            self.callback
                .on_progress(self.budget.used(), self.config.max_turns);

            // Compression check
            if self.config.auto_compress && self.compressor.should_compress() {
                self.hook.on_pre_compact(self.state.messages.len());
                let msgs_clone = self.state.messages.clone();
                if let Ok(result) = self.compressor.compress(msgs_clone, None).await {
                    self.callback
                        .on_compress(result.original_tokens, result.final_tokens);
                    self.hook
                        .on_post_compact(result.original_tokens, result.final_tokens);
                    if let Some(note) = result.note {
                        self.state.set_compression_note(note);
                    }
                    self.state.messages = result.messages;
                    compressed = true;

                    // Post-compact restoration: re-inject deferred tools and active skills
                    self.post_compact_restore();
                }
            }

            // Build request with sorted tool definitions for cache stability
            let mut sdk_tools = self.build_sdk_tools();
            sdk_tools.sort_by(|a, b| a.function.name.cmp(&b.function.name));

            // In Plan mode, filter out non-read-only tools
            if self.config.permission_mode == PermissionMode::Plan {
                sdk_tools.retain(|t| {
                    self.tools
                        .get(&t.function.name)
                        .is_some_and(cold_tools::Tool::is_read_only)
                });
            }

            let mut request =
                ChatRequest::new(&self.config.model, self.state.messages.clone());
            if !sdk_tools.is_empty() {
                request.tools = Some(sdk_tools);
            }

            if self.config.streaming {
                // ── Streaming path ──────────────────────────────
                let stream_result = self
                    .active_client()
                    .chat_stream(&request)
                    .await;

                let mut stream = match stream_result {
                    Ok(s) => s,
                    Err(ref e) if Self::is_context_length_error(e) => {
                        // Try reactive compact recovery
                        if self.try_reactive_recovery()? {
                            continue; // retry with compacted messages
                        }
                        // Cannot unwrap_err because ChatStream is not Debug;
                        // reconstruct the error.
                        return Err(AgentError::Sdk(cold_sdk::ColdError::Api {
                            status: 400,
                            body: "context_length_exceeded".to_string(),
                        }));
                    }
                    Err(e) if !self.using_fallback && self.should_try_fallback(&e) => {
                        self.callback.on_error(&AgentError::Sdk(
                            cold_sdk::ColdError::Config(format!(
                                "primary model failed: {e}, trying fallback"
                            )),
                        ));
                        self.init_fallback_client()?;
                        self.active_client()
                            .chat_stream(&request)
                            .await
                            .map_err(AgentError::Sdk)?
                    }
                    Err(e) => return Err(AgentError::Sdk(e)),
                };

                let mut executor = StreamingToolExecutor::new();

                while let Some(chunk_result) = stream.next().await {
                    let chunk = chunk_result.map_err(AgentError::Sdk)?;

                    if let Some(ref usage) = chunk.usage {
                        self.compressor.update_usage(usage);
                        self.state.total_prompt_tokens += usage.prompt_tokens;
                        self.state.total_completion_tokens += usage.completion_tokens;
                    }

                    for choice in &chunk.choices {
                        if let Some(ref content) = choice.delta.content {
                            self.callback.on_text(content);
                            executor.feed_text(content);
                        }
                        if let Some(ref tc_deltas) = choice.delta.tool_calls {
                            // Eagerly parse arguments during streaming
                            let _ready = executor.feed_and_check(tc_deltas);
                        }
                        if let Some(ref reason) = choice.finish_reason {
                            executor.set_finish_reason(reason.clone());
                        }
                    }
                }

                // Stream finished — process result
                let finish = executor.finish_reason().cloned();

                match finish {
                    Some(FinishReason::ToolCalls) => {
                        // Eager validation: detect malformed JSON before execution
                        let invalid = executor.validate_all_args();
                        if !invalid.is_empty() {
                            self.callback.on_error(&AgentError::Config(
                                format!("malformed tool call arguments at indices: {invalid:?}")
                            ));
                        }

                        // Build assistant message with tool calls
                        let text = executor.take_text();
                        let tcs = executor.take_tool_calls();

                        let assistant_msg = build_assistant_message_with_tool_calls(
                            if text.is_empty() { None } else { Some(&text) },
                            &tcs,
                        );
                        self.state.add_message(assistant_msg.clone());
                        self.append_log_if_configured(Some(assistant_msg)).await;

                        // Execute tools with hook integration
                        let tool_ctx = self.build_tool_context();
                        let exec_results =
                            self.execute_tools_with_hooks(&tcs, &tool_ctx).await;

                        self.record_execution_results(
                            &exec_results,
                            &mut tools_called,
                        )
                        .await;

                        if !self.budget.consume() {
                            break;
                        }
                        self.state.turn_count += 1;
                        self.hook.on_turn_complete(self.state.turn_count);
                        self.dispatcher.reset_for_turn();
                    }
                    Some(FinishReason::Length) => {
                        let partial = executor.take_text();
                        if self.handle_length_recovery(
                            &partial,
                            &mut output_recovery_count,
                            &mut compressed,
                        )
                        .await?
                        {
                            continue;
                        }
                        // Recovery exhausted — return what we have
                        result_text = partial;
                        break;
                    }
                    Some(FinishReason::Stop) => {
                        result_text = executor.take_text();
                        let assistant_msg = ChatMessage::assistant(&result_text);
                        self.state.add_message(assistant_msg.clone());
                        self.append_log_if_configured(Some(assistant_msg)).await;

                        // Hook: on_stop can force continuation
                        if let HookDecision::Block(_) = self.hook.on_stop() {
                            if !self.budget.consume() {
                                break;
                            }
                            continue;
                        }
                        break;
                    }
                    Some(_) | None => {
                        result_text = executor.take_text();
                        let assistant_msg = ChatMessage::assistant(&result_text);
                        self.state.add_message(assistant_msg.clone());
                        self.append_log_if_configured(Some(assistant_msg)).await;
                        break;
                    }
                }
            } else {
                // ── Non-streaming path with fallback ──────────────
                let response = match self.try_with_fallback(&request).await {
                    Ok(r) => r,
                    Err(AgentError::Sdk(ref e)) if Self::is_context_length_error(e) => {
                        if self.try_reactive_recovery()? {
                            continue; // retry with compacted messages
                        }
                        return Err(AgentError::Sdk(
                            cold_sdk::ColdError::Api {
                                status: 400,
                                body: "context_length_exceeded after reactive compact failed"
                                    .to_string(),
                            },
                        ));
                    }
                    Err(e) => return Err(e),
                };

                if let Some(ref usage) = response.usage {
                    self.compressor.update_usage(usage);
                    self.state.total_prompt_tokens += usage.prompt_tokens;
                    self.state.total_completion_tokens += usage.completion_tokens;
                }

                let choice = response
                    .choices
                    .first()
                    .ok_or_else(|| AgentError::Config("empty response from model".into()))?;

                match choice.finish_reason {
                    Some(FinishReason::ToolCalls) => {
                        self.state.add_message(choice.message.clone());
                        self.append_log_if_configured(Some(choice.message.clone()))
                            .await;
                        self.execute_tool_calls(&mut tools_called).await;

                        if !self.budget.consume() {
                            break;
                        }
                        self.state.turn_count += 1;
                        self.hook.on_turn_complete(self.state.turn_count);
                        self.dispatcher.reset_for_turn();
                    }
                    Some(FinishReason::Length) => {
                        let partial = response
                            .text()
                            .unwrap_or_default()
                            .to_string();
                        if self.handle_length_recovery(
                            &partial,
                            &mut output_recovery_count,
                            &mut compressed,
                        )
                        .await?
                        {
                            continue;
                        }
                        result_text = partial;
                        break;
                    }
                    Some(FinishReason::Stop) => {
                        if let Some(text) = response.text() {
                            result_text = text.to_string();
                            self.callback.on_text(text);
                        }
                        self.state.add_message(choice.message.clone());
                        self.append_log_if_configured(Some(choice.message.clone()))
                            .await;

                        // Hook: on_stop can force continuation
                        if let HookDecision::Block(_) = self.hook.on_stop() {
                            if !self.budget.consume() {
                                break;
                            }
                            continue;
                        }
                        break;
                    }
                    None => {
                        if let Some(text) = response.text() {
                            result_text = text.to_string();
                            self.callback.on_text(text);
                        }
                        self.state.add_message(choice.message.clone());
                        self.append_log_if_configured(Some(choice.message.clone()))
                            .await;
                        break;
                    }
                    Some(_) => {
                        if let Some(text) = response.text() {
                            result_text = text.to_string();
                        }
                        self.state.add_message(choice.message.clone());
                        self.append_log_if_configured(Some(choice.message.clone()))
                            .await;
                        break;
                    }
                }
            }
        }

        if !self.budget.has_remaining() && result_text.is_empty() {
            return Err(AgentError::BudgetExhausted {
                turns_used: self.budget.used(),
                max_turns: self.config.max_turns,
            });
        }

        if self.config.persist_session {
            if let Some(ref dir) = self.config.session_dir {
                let _ =
                    save_session(dir, &self.state, &self.config.model, &self.compressor)
                        .await;
            }
        }

        let result = AgentResult {
            text: result_text,
            turns_used: self.budget.used(),
            tokens: TokenUsage {
                prompt_tokens: self.state.total_prompt_tokens,
                completion_tokens: self.state.total_completion_tokens,
                total_tokens: self.state.total_prompt_tokens
                    + self.state.total_completion_tokens,
            },
            tools_called,
            compressed,
        };

        self.callback.on_complete(&result);
        Ok(result)
    }

    /// Handle `FinishReason::Length` with retry recovery.
    ///
    /// Returns `Ok(true)` if the caller should `continue` the loop, `Ok(false)`
    /// if recovery is exhausted and the caller should break.
    async fn handle_length_recovery(
        &mut self,
        partial_text: &str,
        recovery_count: &mut u32,
        compressed: &mut bool,
    ) -> Result<bool, AgentError> {
        const MAX_RECOVERY: u32 = 3;

        if *recovery_count < MAX_RECOVERY {
            *recovery_count += 1;

            // Append partial text as assistant message if non-empty
            if !partial_text.is_empty() {
                let msg = ChatMessage::assistant(partial_text);
                self.state.add_message(msg.clone());
                self.append_log_if_configured(Some(msg)).await;
            }

            // Ask the model to continue
            let cont = ChatMessage::user(
                "Your response was truncated. Please continue from where you left off.",
            );
            self.state.add_message(cont.clone());
            self.append_log_if_configured(Some(cont)).await;

            return Ok(true);
        }

        // Recovery exhausted — try compression as last resort
        if self.config.auto_compress {
            let msgs_clone = self.state.messages.clone();
            if let Ok(result) = self.compressor.compress(msgs_clone, None).await {
                self.state.messages = result.messages;
                *compressed = true;
                return Ok(true);
            }
            return Err(AgentError::Config(
                "context length exceeded and compression failed".into(),
            ));
        }

        Ok(false)
    }

    /// Record execution results into the `tools_called` list and state.
    async fn record_execution_results(
        &mut self,
        exec_results: &[ExecutionResult],
        tools_called: &mut Vec<ToolCallRecord>,
    ) {
        for er in exec_results {
            let (result_content, succeeded) = match &er.result {
                Ok(r) => (r.as_text().to_string(), true),
                Err(e) => (format!("Tool error: {e}"), false),
            };

            tools_called.push(ToolCallRecord {
                name: er.tool_name.clone(),
                args: serde_json::Value::Null,
                result_preview: result_content.chars().take(200).collect(),
                duration_ms: er.duration_ms,
                succeeded,
            });

            let tool_msg = ChatMessage::tool(&er.tool_call_id, &result_content);
            self.state.add_tool_result(&er.tool_call_id, &result_content);
            self.append_log_if_configured(Some(tool_msg)).await;
        }
    }

    /// Execute tool calls with hook integration (for streaming path).
    ///
    /// Pre-call hooks fire for every tool before the batch; blocked tools are
    /// excluded from execution.  The remaining tools are dispatched via
    /// [`Dispatcher::execute_batch`] for true parallel execution, and post-call
    /// hooks fire for each result.
    async fn execute_tools_with_hooks(
        &mut self,
        tool_calls: &[cold_sdk::ToolCall],
        ctx: &ToolContext,
    ) -> Vec<ExecutionResult> {
        let mut results: Vec<Option<ExecutionResult>> =
            (0..tool_calls.len()).map(|_| None).collect();

        // Phase 1: parse args and run pre-call hooks; collect non-blocked calls.
        let mut batch_entries: Vec<(usize, &cold_sdk::ToolCall, serde_json::Value)> =
            Vec::with_capacity(tool_calls.len());

        for (idx, tc) in tool_calls.iter().enumerate() {
            let args: serde_json::Value =
                serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));

            if let HookDecision::Block(reason) = self.hook.on_pre_tool_call(&tc.function.name, &args) {
                let blocked_msg = format!("Blocked by hook: {reason}");
                results[idx] = Some(ExecutionResult {
                    tool_call_id: tc.id.clone(),
                    tool_name: tc.function.name.clone(),
                    result: Ok(cold_tools::ToolResult::error(&blocked_msg, true)),
                    duration_ms: 0,
                });
                continue;
            }

            self.callback.on_tool_call(&tc.function.name, &args);
            batch_entries.push((idx, tc, args));
        }

        // Phase 2: dispatch non-blocked tools through execute_batch.
        if !batch_entries.is_empty() {
            let calls: Vec<(String, serde_json::Value)> = batch_entries
                .iter()
                .map(|(_, tc, args)| (tc.function.name.clone(), args.clone()))
                .collect();

            let start = std::time::Instant::now();
            let batch_results = self.dispatcher.execute_batch(calls, ctx).await;
            #[allow(clippy::cast_possible_truncation)]
            let batch_duration_ms = start.elapsed().as_millis() as u64;
            let per_tool_ms = batch_duration_ms / batch_entries.len() as u64;

            // Phase 3: map results back, fire post-call hooks and callbacks.
            for ((idx, tc, _), result) in batch_entries.into_iter().zip(batch_results) {
                if let Ok(ref r) = result {
                    self.callback.on_tool_result(&tc.function.name, r);
                    self.hook.on_post_tool_call(&tc.function.name, r);
                }

                results[idx] = Some(ExecutionResult {
                    tool_call_id: tc.id.clone(),
                    tool_name: tc.function.name.clone(),
                    result,
                    duration_ms: per_tool_ms,
                });
            }
        }

        results
            .into_iter()
            .map(|r| r.expect("every tool call should have a result"))
            .collect()
    }

    /// Execute all tool calls from the most recent assistant message.
    async fn execute_tool_calls(&mut self, tools_called: &mut Vec<ToolCallRecord>) {
        // Extract tool calls from the last message (which we just pushed)
        let tcs: Vec<cold_sdk::ToolCall> = self
            .state
            .messages
            .last()
            .and_then(|m| m.tool_calls.clone())
            .unwrap_or_default();

        let tool_ctx = self.build_tool_context();

        for tc in &tcs {
            let args: serde_json::Value =
                serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));

            // Hook: pre-tool-call check
            if let HookDecision::Block(reason) = self.hook.on_pre_tool_call(&tc.function.name, &args) {
                let blocked_msg = format!("Blocked by hook: {reason}");
                tools_called.push(ToolCallRecord {
                    name: tc.function.name.clone(),
                    args: args.clone(),
                    result_preview: blocked_msg.clone(),
                    duration_ms: 0,
                    succeeded: false,
                });
                self.state.add_tool_result(&tc.id, &blocked_msg);
                let tool_msg = ChatMessage::tool(&tc.id, &blocked_msg);
                self.append_log_if_configured(Some(tool_msg)).await;
                continue;
            }

            self.callback.on_tool_call(&tc.function.name, &args);

            let start = std::time::Instant::now();
            let tool_result = self
                .dispatcher
                .execute_one(&tc.function.name, args.clone(), &tool_ctx)
                .await;
            #[allow(clippy::cast_possible_truncation)]
            let duration_ms = start.elapsed().as_millis() as u64;

            let (result_content, succeeded) = match tool_result {
                Ok(ref r) => {
                    self.callback.on_tool_result(&tc.function.name, r);
                    self.hook.on_post_tool_call(&tc.function.name, r);
                    (r.as_text().to_string(), true)
                }
                Err(ref e) => (format!("Tool error: {e}"), false),
            };

            tools_called.push(ToolCallRecord {
                name: tc.function.name.clone(),
                args: args.clone(),
                result_preview: result_content.chars().take(200).collect(),
                duration_ms,
                succeeded,
            });

            self.state.add_tool_result(&tc.id, &result_content);

            let tool_msg = ChatMessage::tool(&tc.id, &result_content);
            self.append_log_if_configured(Some(tool_msg)).await;
        }
    }

    /// Convert the tool registry definitions to `cold_sdk::Tool` format.
    fn build_sdk_tools(&self) -> Vec<SdkTool> {
        self.tools
            .get_definitions()
            .into_iter()
            .filter_map(|def| {
                let func = def.get("function")?;
                let name = func.get("name")?.as_str()?.to_string();
                let description =
                    func.get("description").and_then(|v| v.as_str()).map(String::from);
                let parameters = func.get("parameters").cloned();
                Some(SdkTool {
                    tool_type: "function".to_string(),
                    function: cold_sdk::FunctionDef {
                        name,
                        description,
                        parameters,
                        strict: None,
                    },
                })
            })
            .collect()
    }

    /// Build a `ToolContext` for the current execution environment.
    fn build_tool_context(&self) -> ToolContext {
        ToolContext {
            cwd: self.config.root_dir.clone(),
            root: self.config.root_dir.clone(),
            task_id: self.state.session_id.clone(),
            user: Arc::new(AutoApprove),
            cancelled: Arc::clone(&self.cancelled),
            env: HashMap::new(),
            plan_mode: Arc::new(AtomicBool::new(
                self.config.permission_mode == PermissionMode::Plan,
            )),
        }
    }
}

/// Build an assistant message that includes both text content and tool calls.
fn build_assistant_message_with_tool_calls(
    text: Option<&str>,
    tool_calls: &[cold_sdk::ToolCall],
) -> ChatMessage {
    ChatMessage {
        role: cold_sdk::Role::Assistant,
        content: text.map(|t| cold_sdk::MessageContent::Text(t.to_string())),
        name: None,
        tool_calls: Some(tool_calls.to_vec()),
        tool_call_id: None,
        refusal: None,
    }
}
