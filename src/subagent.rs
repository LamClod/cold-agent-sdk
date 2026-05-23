/// Sub-agent system for delegated task execution.
///
/// A sub-agent runs with its own conversation state but shares the parent's
/// API client and tool registry. This enables the parent agent to spawn
/// isolated workers for specific goals.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cold_sdk::{ChatMessage, ChatRequest, ColdClient, FinishReason, Tool as SdkTool};
use cold_tools::{AutoApprove, Dispatcher, PermissionMode, ToolContext, ToolRegistry};
use serde_json::json;

use crate::budget::IterationBudget;
use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::result::{AgentResult, TokenUsage, ToolCallRecord};
use crate::state::ConversationState;

/// Configuration for a sub-agent invocation.
pub struct SubAgentConfig {
    /// The goal the sub-agent should accomplish.
    pub goal: String,
    /// Optional additional context for the sub-agent.
    pub context: Option<String>,
    /// Maximum turns for the sub-agent (default 30, less than parent's 90).
    pub max_turns: u32,
}

impl Default for SubAgentConfig {
    fn default() -> Self {
        Self {
            goal: String::new(),
            context: None,
            max_turns: 30,
        }
    }
}

/// Built-in agent type presets.
pub enum AgentType {
    /// General purpose with full tool access.
    General,
    /// Explore mode: read-only tools, no writes.
    Explore,
    /// Plan mode: read-only + think, outputs a structured plan.
    Plan,
}

/// Run a sub-agent with isolated context.
///
/// The sub-agent:
/// 1. Gets its own conversation state (fresh messages).
/// 2. Shares the same `ColdClient` and `ToolRegistry`.
/// 3. Has a reduced iteration budget.
/// 4. Gets a system prompt with the goal injected.
/// 5. Returns its result to the parent agent.
///
/// # Errors
///
/// Returns [`AgentError`] if the sub-agent loop fails.
#[allow(clippy::too_many_lines)]
pub async fn run_subagent(
    parent_config: &AgentConfig,
    client: &ColdClient,
    tools: Arc<ToolRegistry>,
    sub_config: SubAgentConfig,
    agent_type: AgentType,
) -> Result<AgentResult, AgentError> {
    // Build the sub-agent system prompt.
    let system_prompt = build_subagent_prompt(&sub_config, &agent_type);

    // Determine permission mode based on agent type.
    let permission_mode = match agent_type {
        AgentType::General => parent_config.permission_mode,
        AgentType::Explore | AgentType::Plan => PermissionMode::Plan,
    };

    // Create fresh state and budget.
    let mut state = ConversationState::new();
    let mut budget = IterationBudget::new(sub_config.max_turns);
    let mut dispatcher = Dispatcher::new(Arc::clone(&tools));
    let cancelled = Arc::new(AtomicBool::new(false));
    // Callback unused for sub-agents (they run silently).

    // Seed the conversation.
    state.add_message(ChatMessage::system(&system_prompt));
    state.add_message(ChatMessage::user(&sub_config.goal));

    // Run the loop.
    let mut result_text = String::new();
    let mut tools_called: Vec<ToolCallRecord> = Vec::new();

    while budget.has_remaining() {
        if cancelled.load(Ordering::Relaxed) {
            return Err(AgentError::Interrupted);
        }

        // Build tool definitions, filtered by permission mode.
        let mut sdk_tools = build_sdk_tools(&tools);
        sdk_tools.sort_by(|a, b| a.function.name.cmp(&b.function.name));

        if permission_mode == PermissionMode::Plan {
            sdk_tools.retain(|t| {
                tools
                    .get(&t.function.name)
                    .is_some_and(cold_tools::Tool::is_read_only)
            });
        }

        let mut request =
            ChatRequest::new(&parent_config.model, state.messages.clone());
        if !sdk_tools.is_empty() {
            request.tools = Some(sdk_tools);
        }

        // Non-streaming path for sub-agents (simpler, reliable).
        let response = client.chat(&request).await.map_err(AgentError::Sdk)?;

        if let Some(ref usage) = response.usage {
            state.total_prompt_tokens += usage.prompt_tokens;
            state.total_completion_tokens += usage.completion_tokens;
        }

        let choice = response
            .choices
            .first()
            .ok_or_else(|| AgentError::Config("empty response from model".into()))?;

        if choice.finish_reason == Some(FinishReason::ToolCalls) {
            state.add_message(choice.message.clone());

            let tcs: Vec<cold_sdk::ToolCall> = choice
                .message
                .tool_calls
                .clone()
                .unwrap_or_default();

            let tool_ctx = ToolContext {
                cwd: parent_config.root_dir.clone(),
                root: parent_config.root_dir.clone(),
                task_id: state.session_id.clone(),
                user: Arc::new(AutoApprove),
                cancelled: Arc::clone(&cancelled),
                env: HashMap::new(),
                plan_mode: Arc::new(AtomicBool::new(
                    permission_mode == PermissionMode::Plan,
                )),
            };

            for tc in &tcs {
                let args: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments)
                        .unwrap_or_else(|_| json!({}));

                let start = std::time::Instant::now();
                let tool_result = dispatcher
                    .execute_one(&tc.function.name, args.clone(), &tool_ctx)
                    .await;
                #[allow(clippy::cast_possible_truncation)]
                let duration_ms = start.elapsed().as_millis() as u64;

                let (result_content, succeeded) = match &tool_result {
                    Ok(r) => (r.as_text().to_string(), true),
                    Err(e) => (format!("Tool error: {e}"), false),
                };

                tools_called.push(ToolCallRecord {
                    name: tc.function.name.clone(),
                    args: args.clone(),
                    result_preview: result_content.chars().take(200).collect(),
                    duration_ms,
                    succeeded,
                });

                state.add_tool_result(&tc.id, &result_content);
            }

            if !budget.consume() {
                break;
            }
            state.turn_count += 1;
            dispatcher.reset_for_turn();
        } else {
            if let Some(text) = response.text() {
                result_text = text.to_string();
            }
            state.add_message(choice.message.clone());
            break;
        }
    }

    if !budget.has_remaining() && result_text.is_empty() {
        return Err(AgentError::BudgetExhausted {
            turns_used: budget.used(),
            max_turns: sub_config.max_turns,
        });
    }

    Ok(AgentResult {
        text: result_text,
        turns_used: budget.used(),
        tokens: TokenUsage {
            prompt_tokens: state.total_prompt_tokens,
            completion_tokens: state.total_completion_tokens,
            total_tokens: state.total_prompt_tokens + state.total_completion_tokens,
        },
        tools_called,
        compressed: false,
    })
}

/// Build the system prompt for a sub-agent.
fn build_subagent_prompt(config: &SubAgentConfig, agent_type: &AgentType) -> String {
    let mut parts = vec![
        "You are a sub-agent. Complete the assigned goal efficiently and report your findings.".to_string(),
    ];

    parts.push(format!("Your goal: {}", config.goal));

    if let Some(ref ctx) = config.context {
        parts.push(format!("Context: {ctx}"));
    }

    match agent_type {
        AgentType::General => {
            parts.push(
                "You have full tool access. Use tools to accomplish the goal.".to_string(),
            );
        }
        AgentType::Explore => {
            parts.push(
                "You are in EXPLORE mode. You can read files and search, but cannot modify anything. Report your findings.".to_string(),
            );
        }
        AgentType::Plan => {
            parts.push(
                "You are in PLAN mode. Analyze the codebase and output a structured plan with clear steps. Do not make any changes.".to_string(),
            );
        }
    }

    parts.join("\n\n")
}

/// Convert tool registry definitions to SDK tool format.
fn build_sdk_tools(tools: &ToolRegistry) -> Vec<SdkTool> {
    tools
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_subagent_prompt_general() {
        let config = SubAgentConfig {
            goal: "Find all TODO comments".into(),
            context: Some("In the src/ directory".into()),
            max_turns: 10,
        };
        let prompt = build_subagent_prompt(&config, &AgentType::General);
        assert!(prompt.contains("sub-agent"));
        assert!(prompt.contains("Find all TODO comments"));
        assert!(prompt.contains("In the src/ directory"));
        assert!(prompt.contains("full tool access"));
    }

    #[test]
    fn test_build_subagent_prompt_plan() {
        let config = SubAgentConfig {
            goal: "Plan refactoring".into(),
            context: None,
            max_turns: 30,
        };
        let prompt = build_subagent_prompt(&config, &AgentType::Plan);
        assert!(prompt.contains("PLAN mode"));
        assert!(!prompt.contains("Context:"));
    }

    #[test]
    fn test_subagent_config_default() {
        let config = SubAgentConfig::default();
        assert_eq!(config.max_turns, 30);
        assert!(config.goal.is_empty());
        assert!(config.context.is_none());
    }
}
