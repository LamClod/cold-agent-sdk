/// Delegate tool: allows the agent to spawn sub-agents via tool calling.
///
/// This tool is registered in the agent's tool registry and invokes
/// [`run_subagent`](crate::subagent::run_subagent) when executed.
use std::pin::Pin;
use std::sync::Arc;

use cold_sdk::ColdClient;
use cold_tools::error::ToolError;
use cold_tools::tool::{Permission, Tool, ToolResult};
use cold_tools::{CoreToolConfig, ToolRegistry};
use serde_json::{Value, json};

use crate::config::AgentConfig;
use crate::subagent::{AgentType, SubAgentConfig, run_subagent};

/// A tool that delegates work to a sub-agent.
///
/// Holds the parent agent's client and config snapshot so the sub-agent
/// can reuse the same API connection. A fresh `ToolRegistry` is built for
/// each sub-agent invocation to avoid circular `Arc` references.
pub struct DelegateTool {
    client: ColdClient,
    model: String,
    api_key: String,
    base_url: Option<String>,
    root_dir: std::path::PathBuf,
    permission_mode: cold_tools::PermissionMode,
}

impl DelegateTool {
    /// Create a new delegate tool from parent agent resources.
    #[must_use]
    pub fn new(client: ColdClient, config: &AgentConfig) -> Self {
        Self {
            client,
            model: config.model.clone(),
            api_key: config.api_key.clone(),
            base_url: config.base_url.clone(),
            root_dir: config.root_dir.clone(),
            permission_mode: config.permission_mode,
        }
    }
}

impl Tool for DelegateTool {
    fn name(&self) -> &'static str {
        "delegate"
    }

    fn description(&self) -> &'static str {
        "Delegate a task to a sub-agent that runs with its own context. \
         Use this for complex sub-tasks that benefit from a fresh conversation, \
         such as exploring a large codebase or planning a multi-step change."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What the sub-agent should accomplish."
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the sub-agent."
                },
                "agent_type": {
                    "type": "string",
                    "enum": ["general", "explore", "plan"],
                    "description": "Agent type: 'general' (full access), 'explore' (read-only), 'plan' (read-only + structured plan output). Defaults to 'general'."
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum turns for the sub-agent. Defaults to 30."
                }
            },
            "required": ["goal"]
        })
    }

    fn permission(&self) -> Permission {
        Permission::Auto
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn timeout_secs(&self) -> u64 {
        600 // Sub-agents can take a while.
    }

    fn execute<'a>(
        &'a self,
        args: Value,
        _ctx: &'a cold_tools::ToolContext,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult, ToolError>> + Send + 'a>>
    {
        Box::pin(async move {
            let goal = args
                .get("goal")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            if goal.is_empty() {
                return Ok(ToolResult::error("'goal' parameter is required", true));
            }

            let context = args
                .get("context")
                .and_then(Value::as_str)
                .map(String::from);

            let agent_type = match args
                .get("agent_type")
                .and_then(Value::as_str)
                .unwrap_or("general")
            {
                "explore" => AgentType::Explore,
                "plan" => AgentType::Plan,
                _ => AgentType::General,
            };

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let max_turns = args
                .get("max_turns")
                .and_then(Value::as_u64)
                .map_or(30, |v| v as u32);

            let sub_config = SubAgentConfig {
                goal,
                context,
                max_turns,
            };

            // Build a minimal parent config for the sub-agent.
            let parent_config = AgentConfig::new(
                &self.model,
                128_000, // reasonable default
                &self.api_key,
            )
            .with_root_dir(&self.root_dir)
            .with_permission_mode(self.permission_mode);

            let parent_config = if let Some(ref url) = self.base_url {
                parent_config.with_base_url(url)
            } else {
                parent_config
            };

            // Build a fresh tool registry for the sub-agent.
            let mut sub_tools = ToolRegistry::new();
            cold_tools::register_core_tools(
                &mut sub_tools,
                CoreToolConfig {
                    root_dir: self.root_dir.clone(),
                    ..CoreToolConfig::default()
                },
            );
            let sub_tools = Arc::new(sub_tools);

            match run_subagent(
                &parent_config,
                &self.client,
                sub_tools,
                sub_config,
                agent_type,
            )
            .await
            {
                Ok(result) => {
                    let summary = format!(
                        "Sub-agent completed in {} turns ({} prompt + {} completion tokens).\n\n{}",
                        result.turns_used,
                        result.tokens.prompt_tokens,
                        result.tokens.completion_tokens,
                        result.text
                    );
                    Ok(ToolResult::text(summary))
                }
                Err(e) => Ok(ToolResult::error(
                    format!("Sub-agent failed: {e}"),
                    true,
                )),
            }
        })
    }
}
