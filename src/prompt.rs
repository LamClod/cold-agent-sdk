use std::path::Path;

use cold_tools::PermissionMode;

use crate::config::AgentConfig;
use crate::memory::{build_memory_prompt, load_memory_files};
use crate::skill::SkillRegistry;
use crate::state::ConversationState;

/// Default identity prompt injected when no custom system prompt is set.
pub const DEFAULT_IDENTITY: &str = r"You are a powerful AI coding assistant powered by LAMCLOD. You help users with software engineering tasks including writing code, debugging, refactoring, and more.";

/// Guidance for effective tool usage.
pub const TOOL_USE_GUIDANCE: &str = r"# Using your tools
- Use dedicated tools instead of describing what you would do. Actually DO it.
- Read files before modifying them to understand existing code.
- Use search_files to find relevant code before making changes.
- Break complex tasks into steps. Execute one step at a time and verify results.
- If a tool call fails, analyze the error before retrying with a different approach.
- Use think tool to plan complex multi-step operations before executing.
- Call multiple tools in a single response when they are independent (no dependencies between them).
- Reserve terminal/bash for commands that require shell execution. Use dedicated file tools for file operations.";

/// Guidance for code style and minimal changes.
pub const CODE_STYLE_GUIDANCE: &str = r"# Code style
- Don't add features, refactor code, or make improvements beyond what was asked.
- A bug fix doesn't need surrounding code cleaned up.
- Don't add docstrings, comments, or type annotations to code you didn't change.
- Only add comments where logic isn't self-evident.
- Three similar lines of code is better than a premature abstraction.
- Don't create helpers or utilities for one-time operations.";

/// Guidance for safe and reversible actions.
pub const ACTION_SAFETY_GUIDANCE: &str = r"# Executing actions with care
Consider the reversibility and blast radius of actions:
- Freely take local, reversible actions (editing files, running tests).
- For hard-to-reverse actions (deleting files, force-pushing, sending messages), confirm with the user first.
- Never use destructive operations as shortcuts. Investigate root causes instead.
- If you discover unexpected state (unfamiliar files, branches), investigate before overwriting.";

/// Guidance for concise, efficient output.
pub const OUTPUT_EFFICIENCY: &str = r"# Output efficiency
- Go straight to the point. Lead with the answer, not the reasoning.
- Skip filler words, preamble, and transitions.
- Do not restate what the user said.
- If you can say it in one sentence, don't use three.
- Focus text output on: decisions needing input, status updates at milestones, errors that change the plan.";

/// Guidance injected when the agent is in plan (read-only) mode.
pub const PLAN_MODE_GUIDANCE: &str = "You are in PLAN MODE. You can read files, search, and analyze, but cannot modify files or execute commands. Describe what changes you would make and why.";

/// Guidance for hook behavior.
pub const HOOKS_GUIDANCE: &str = r"# Hooks
When hooks are configured, they run automatically at lifecycle events:
- Before tool execution (can block specific operations)
- After tool execution (for logging/side effects)
- Before/after context compression
- At session start and stop
Do not attempt to replicate hook behavior manually.";

/// Guidance for language adaptation.
pub const LANGUAGE_GUIDANCE: &str = r"# Language
Respond in the same language the user uses. If the user writes in Chinese, respond in Chinese. If in English, respond in English. Technical terms may remain in English.";

/// Guidance for output formatting.
pub const OUTPUT_STYLE_GUIDANCE: &str = r"# Output style
- Use GitHub-flavored Markdown for formatting
- When referencing code, include file:line format (e.g. src/main.rs:42)
- Use code blocks with language tags for code snippets
- Keep inline code for identifiers and short expressions";

/// Guidance for MCP tool usage.
pub const MCP_GUIDANCE: &str = r"# MCP Tools
Some tools are provided by external MCP servers. These tools have the same interface as built-in tools but may have different latency and error characteristics. Treat MCP tool errors as potentially transient.";

/// Guidance for the think/scratchpad tool.
pub const SCRATCHPAD_GUIDANCE: &str = r"# Thinking
Use the think tool to plan complex multi-step operations before executing. The think tool content is visible in your context but not shown to the user. Use it for:
- Breaking down complex tasks into steps
- Analyzing errors before deciding on fixes
- Weighing tradeoffs between approaches";

/// Boundary marker between static (cacheable) and dynamic prompt sections.
///
/// API consumers can use this marker to place cache control breakpoints,
/// ensuring that the static prefix is cached across requests while the
/// dynamic suffix varies per turn.
pub const CACHE_BOUNDARY: &str = "__CACHE_BOUNDARY__";

/// Split a system prompt at the cache boundary.
///
/// Returns `(static_part, dynamic_part)`.  If no boundary is found, returns
/// `(full_prompt, "")`.
///
/// API consumers (e.g. Anthropic-specific integrations) can use this to place
/// `cache_control` breakpoints so the static prefix is cached across requests
/// while the dynamic suffix varies per turn.
#[must_use]
pub fn split_at_cache_boundary(prompt: &str) -> (&str, &str) {
    prompt.find(CACHE_BOUNDARY).map_or((prompt, ""), |pos| {
        let static_part = prompt[..pos].trim_end();
        let dynamic_part = prompt[pos + CACHE_BOUNDARY.len()..].trim_start();
        (static_part, dynamic_part)
    })
}

/// Strip the [`CACHE_BOUNDARY`] marker from a prompt before sending it to the
/// model.  The marker is a client-side signal and should not appear in the
/// text the model receives.
#[must_use]
pub fn strip_cache_boundary(prompt: &str) -> String {
    prompt.replace(CACHE_BOUNDARY, "")
}

/// Maximum bytes to read from a project context file.
const MAX_PROJECT_CONTEXT_BYTES: u64 = 50 * 1024;

/// Assemble the full system prompt from all available sources.
///
/// Layout (static sections are cache-friendly):
/// 1. **Identity** — custom or `DEFAULT_IDENTITY`
/// 2. **Tool-use guidance**
/// 3. **Code style guidance**
/// 4. **Action safety guidance**
/// 5. **Output efficiency**
/// 6. `CACHE_BOUNDARY`
/// 7. **Skills injection** (if any match)
/// 8. **Plan mode guidance** (if plan mode)
/// 9. **Hooks guidance** (if hooks configured)
/// 10. **Language guidance** (always)
/// 11. **Output style guidance** (always)
/// 12. **MCP guidance** (if MCP tools registered)
/// 13. **Scratchpad guidance** (always)
/// 14. **Memory injection** (if memory files found)
/// 15. **Project context** (`.cold.md`)
/// 16. **Date + model info**
/// 17. **Compression note** (if compressed)
#[must_use]
pub fn build_system_prompt(
    config: &AgentConfig,
    skills: &SkillRegistry,
    state: &ConversationState,
    has_hooks: bool,
    has_mcp_tools: bool,
) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let mut owned: Vec<String> = Vec::new();

    // ── Static sections (before cache boundary) ──

    // 1. Identity
    let identity = config
        .system_prompt
        .as_deref()
        .unwrap_or(DEFAULT_IDENTITY);
    parts.push(identity);

    // 2. Tool-use guidance
    parts.push(TOOL_USE_GUIDANCE);

    // 3. Code style guidance
    parts.push(CODE_STYLE_GUIDANCE);

    // 4. Action safety guidance
    parts.push(ACTION_SAFETY_GUIDANCE);

    // 5. Output efficiency
    parts.push(OUTPUT_EFFICIENCY);

    // Collect static parts
    let mut all: Vec<&str> = parts;

    // ── Cache boundary ──
    all.push(CACHE_BOUNDARY);

    // ── Dynamic sections (after cache boundary) ──

    // 7. Skills injection
    let skill_injection = skills.build_prompt_injection(&state.last_user_message);
    if let Some(ref injection) = skill_injection {
        owned.push(injection.clone());
    }

    // 8. Plan mode guidance (dynamic — can change per session)
    if config.permission_mode == PermissionMode::Plan {
        owned.push(PLAN_MODE_GUIDANCE.to_string());
    }

    // 9. Hooks guidance (if hooks are configured)
    if has_hooks {
        owned.push(HOOKS_GUIDANCE.to_string());
    }

    // 10. Language guidance (always)
    owned.push(LANGUAGE_GUIDANCE.to_string());

    // 11. Output style guidance (always)
    owned.push(OUTPUT_STYLE_GUIDANCE.to_string());

    // 12. MCP guidance (if MCP tools registered)
    if has_mcp_tools {
        owned.push(MCP_GUIDANCE.to_string());
    }

    // 13. Scratchpad guidance (always)
    owned.push(SCRATCHPAD_GUIDANCE.to_string());

    // 14. Memory injection
    let memory_entries = load_memory_files(&config.root_dir);
    let memory_prompt = build_memory_prompt(&memory_entries);
    if !memory_prompt.is_empty() {
        owned.push(memory_prompt);
    }

    // 15. Project context
    if let Some(ctx) = load_project_context(&config.root_dir) {
        owned.push(ctx);
    }

    // 16. Volatile: date + model
    owned.push(format!(
        "Current date: {}. Model: {}.",
        current_date_string(),
        config.model
    ));

    // 17. Compression note
    if let Some(ref note) = state.compression_note {
        owned.push(note.clone());
    }

    for s in &owned {
        all.push(s);
    }

    all.join("\n\n")
}

/// Look for `.cold.md` or `COLD.md` in the given directory and return its
/// contents, capped at 50 KB.
#[must_use]
pub fn load_project_context(root_dir: &Path) -> Option<String> {
    for name in &[".cold.md", "COLD.md"] {
        let path = root_dir.join(name);
        if path.is_file() {
            if let Ok(meta) = std::fs::metadata(&path) {
                if meta.len() > MAX_PROJECT_CONTEXT_BYTES {
                    // Read only the first 50 KB
                    if let Ok(bytes) = std::fs::read(&path) {
                        #[allow(clippy::cast_possible_truncation)]
                        let end = MAX_PROJECT_CONTEXT_BYTES as usize;
                        let slice = &bytes[..end.min(bytes.len())];
                        return String::from_utf8_lossy(slice)
                            .into_owned()
                            .into();
                    }
                } else if let Ok(content) = std::fs::read_to_string(&path) {
                    return Some(content);
                }
            }
        }
    }
    None
}

fn current_date_string() -> String {
    // Use a simple UTC date without pulling in chrono.
    let now = std::time::SystemTime::now();
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Rough date calculation
    let days = secs / 86400;
    let year = 1970 + (days * 400 / 146_097);
    format!("{year} (approx)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_project_context_missing() {
        let result = load_project_context(Path::new("/nonexistent/path"));
        assert!(result.is_none());
    }

    #[test]
    fn test_build_system_prompt_default() {
        let config = AgentConfig::new("test-model", 128_000, "sk-test");
        let skills = SkillRegistry::new();
        let state = ConversationState::new();
        let prompt = build_system_prompt(&config, &skills, &state, false, false);
        assert!(prompt.contains("LAMCLOD"));
        assert!(prompt.contains("Using your tools"));
        assert!(prompt.contains("Code style"));
        assert!(prompt.contains("Executing actions with care"));
        assert!(prompt.contains("Output efficiency"));
        assert!(prompt.contains("test-model"));
        assert!(prompt.contains(CACHE_BOUNDARY));
        // New guidance sections (always-on)
        assert!(prompt.contains(LANGUAGE_GUIDANCE));
        assert!(prompt.contains(OUTPUT_STYLE_GUIDANCE));
        assert!(prompt.contains(SCRATCHPAD_GUIDANCE));
        // Should NOT contain hooks/MCP guidance when flags are false
        assert!(!prompt.contains(HOOKS_GUIDANCE));
        assert!(!prompt.contains(MCP_GUIDANCE));
    }

    #[test]
    fn test_build_system_prompt_with_hooks_and_mcp() {
        let config = AgentConfig::new("test-model", 128_000, "sk-test");
        let skills = SkillRegistry::new();
        let state = ConversationState::new();
        let prompt = build_system_prompt(&config, &skills, &state, true, true);
        assert!(prompt.contains(HOOKS_GUIDANCE));
        assert!(prompt.contains(MCP_GUIDANCE));
    }

    #[test]
    fn test_build_system_prompt_plan_mode() {
        let config = AgentConfig::new("test-model", 128_000, "sk-test")
            .with_permission_mode(PermissionMode::Plan);
        let skills = SkillRegistry::new();
        let state = ConversationState::new();
        let prompt = build_system_prompt(&config, &skills, &state, false, false);
        assert!(prompt.contains(PLAN_MODE_GUIDANCE));
    }

    #[test]
    fn test_split_at_cache_boundary() {
        let prompt = format!("static part\n\n{CACHE_BOUNDARY}\n\ndynamic part");
        let (s, d) = split_at_cache_boundary(&prompt);
        assert_eq!(s, "static part");
        assert_eq!(d, "dynamic part");
    }

    #[test]
    fn test_split_at_cache_boundary_no_marker() {
        let (s, d) = split_at_cache_boundary("no marker here");
        assert_eq!(s, "no marker here");
        assert_eq!(d, "");
    }

    #[test]
    fn test_strip_cache_boundary() {
        let prompt = format!("before{CACHE_BOUNDARY}after");
        let stripped = strip_cache_boundary(&prompt);
        assert!(!stripped.contains(CACHE_BOUNDARY));
        assert_eq!(stripped, "beforeafter");
    }

    #[test]
    fn test_cache_boundary_position() {
        let config = AgentConfig::new("test-model", 128_000, "sk-test");
        let skills = SkillRegistry::new();
        let state = ConversationState::new();
        let prompt = build_system_prompt(&config, &skills, &state, false, false);
        let boundary_pos = prompt.find(CACHE_BOUNDARY).expect("boundary must exist");
        let model_pos = prompt.find("test-model").expect("model must exist");
        // Dynamic content (model) must come after the boundary
        assert!(model_pos > boundary_pos);
    }
}
