//! Comprehensive integration tests for cold-agent-sdk.
//!
//! Covers budget, state, prompt, skill, session, memory, hooks, callbacks,
//! config, and streaming executor subsystems.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cold_agent_sdk::*;
use cold_sdk::{ChatMessage, FinishReason, StreamFunctionCall, StreamToolCall};

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

/// Create a unique temporary directory for test isolation.
fn temp_dir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cold_agent_test_{prefix}_{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Remove a temporary directory (best-effort).
fn cleanup(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}

// ═══════════════════════════════════════════════════════════════
// 1. Budget Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn budget_exhausts_after_max_turns() {
    let mut budget = IterationBudget::new(3);

    // Consume 3 normal turns
    assert!(budget.consume());
    assert!(budget.consume());
    assert!(budget.consume());

    // Grace turn still available
    assert!(budget.has_remaining());
    assert!(budget.consume()); // grace

    // Now fully exhausted
    assert!(!budget.has_remaining());
    assert!(!budget.consume());
}

#[test]
fn budget_grace_turn() {
    let mut budget = IterationBudget::new(2);

    // Exhaust normal turns
    assert!(budget.consume());
    assert!(budget.consume());

    // Normal remaining is 0
    assert_eq!(budget.remaining(), 0);

    // But has_remaining still true (grace)
    assert!(budget.has_remaining());

    // Grace turn succeeds
    assert!(budget.consume());
    assert_eq!(budget.used(), 3); // 2 normal + 1 grace

    // Now truly done
    assert!(!budget.has_remaining());
}

#[test]
fn budget_refund() {
    let mut budget = IterationBudget::new(5);

    budget.consume();
    budget.consume();
    assert_eq!(budget.used(), 2);
    assert_eq!(budget.remaining(), 3);

    budget.refund();
    assert_eq!(budget.used(), 1);
    assert_eq!(budget.remaining(), 4);

    // Refund at 0 is safe (no underflow)
    budget.refund();
    assert_eq!(budget.used(), 0);
    budget.refund(); // should be no-op
    assert_eq!(budget.used(), 0);
}

// ═══════════════════════════════════════════════════════════════
// 2. State Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn state_add_messages() {
    let mut state = ConversationState::new();

    state.add_message(ChatMessage::system("You are helpful"));
    state.add_message(ChatMessage::user("Hello"));
    state.add_message(ChatMessage::assistant("Hi there"));
    state.add_message(ChatMessage::tool("call_1", "result data"));

    assert_eq!(state.messages.len(), 4);
    assert_eq!(state.messages[0].role, cold_sdk::Role::System);
    assert_eq!(state.messages[1].role, cold_sdk::Role::User);
    assert_eq!(state.messages[2].role, cold_sdk::Role::Assistant);
    assert_eq!(state.messages[3].role, cold_sdk::Role::Tool);
}

#[test]
fn state_take_and_restore() {
    let mut state = ConversationState::new();

    state.add_message(ChatMessage::user("msg1"));
    state.add_message(ChatMessage::assistant("msg2"));
    state.add_message(ChatMessage::user("msg3"));

    // Take empties the state
    let taken = state.take_messages();
    assert_eq!(taken.len(), 3);
    assert!(state.messages.is_empty());

    // Restore puts them back
    state.restore_messages(taken);
    assert_eq!(state.messages.len(), 3);
}

#[test]
fn state_token_tracking() {
    let mut state = ConversationState::new();

    assert_eq!(state.total_prompt_tokens, 0);
    assert_eq!(state.total_completion_tokens, 0);

    state.total_prompt_tokens = 1500;
    state.total_completion_tokens = 500;

    assert_eq!(state.total_prompt_tokens, 1500);
    assert_eq!(state.total_completion_tokens, 500);
}

#[test]
fn state_add_tool_result() {
    let mut state = ConversationState::new();

    state.add_tool_result("call_42", "some tool output");

    assert_eq!(state.messages.len(), 1);
    assert_eq!(state.messages[0].role, cold_sdk::Role::Tool);
    assert_eq!(
        state.messages[0].tool_call_id.as_deref(),
        Some("call_42")
    );
}

#[test]
fn state_session_id_unique() {
    let s1 = ConversationState::new();
    let s2 = ConversationState::new();
    assert_ne!(s1.session_id, s2.session_id);
}

#[test]
fn state_compression_note() {
    let mut state = ConversationState::new();
    assert!(state.compression_note.is_none());

    state.set_compression_note("Compressed 50k -> 10k tokens".to_string());
    assert_eq!(
        state.compression_note.as_deref(),
        Some("Compressed 50k -> 10k tokens")
    );
}

// ═══════════════════════════════════════════════════════════════
// 3. Prompt Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn prompt_includes_identity() {
    let config = AgentConfig::new("test-model", 128_000, "sk-test");
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let prompt = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );

    // Default identity contains "LAMCLOD"
    assert!(prompt.contains("LAMCLOD"));
}

#[test]
fn prompt_custom_identity() {
    let config = AgentConfig::new("test-model", 128_000, "sk-test")
        .with_system_prompt("I am CustomBot.");
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let prompt = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );

    assert!(prompt.contains("I am CustomBot."));
    // Default identity should NOT be present
    assert!(!prompt.contains("LAMCLOD"));
}

#[test]
fn prompt_includes_guidance_sections() {
    let config = AgentConfig::new("test-model", 128_000, "sk-test");
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let prompt = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );

    assert!(prompt.contains("Using your tools"));
    assert!(prompt.contains("Code style"));
    assert!(prompt.contains("Executing actions with care"));
    assert!(prompt.contains("Output efficiency"));
    assert!(prompt.contains(LANGUAGE_GUIDANCE));
    assert!(prompt.contains(OUTPUT_STYLE_GUIDANCE));
    assert!(prompt.contains(SCRATCHPAD_GUIDANCE));
}

#[test]
fn prompt_plan_mode_guidance() {
    use cold_tools::PermissionMode;

    let config = AgentConfig::new("test-model", 128_000, "sk-test")
        .with_permission_mode(PermissionMode::Plan);
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let prompt = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );

    assert!(prompt.contains("PLAN MODE"));
}

#[test]
fn prompt_hooks_guidance_when_active() {
    let config = AgentConfig::new("test-model", 128_000, "sk-test");
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    // With hooks active
    let with_hooks = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, true, false,
    );
    assert!(with_hooks.contains(HOOKS_GUIDANCE));

    // Without hooks
    let without_hooks = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );
    assert!(!without_hooks.contains(HOOKS_GUIDANCE));
}

#[test]
fn prompt_mcp_guidance_when_active() {
    let config = AgentConfig::new("test-model", 128_000, "sk-test");
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let with_mcp = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, true,
    );
    assert!(with_mcp.contains(MCP_GUIDANCE));

    let without_mcp = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );
    assert!(!without_mcp.contains(MCP_GUIDANCE));
}

#[test]
fn prompt_cache_boundary_present() {
    let config = AgentConfig::new("test-model", 128_000, "sk-test");
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let prompt = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );

    assert!(prompt.contains(CACHE_BOUNDARY));
}

#[test]
fn prompt_strip_cache_boundary() {
    let input = format!("before {} after", CACHE_BOUNDARY);
    let stripped = strip_cache_boundary(&input);

    assert!(!stripped.contains(CACHE_BOUNDARY));
    assert!(stripped.contains("before"));
    assert!(stripped.contains("after"));
}

#[test]
fn prompt_split_cache_boundary() {
    let prompt = format!("static part\n\n{}\n\ndynamic part", CACHE_BOUNDARY);
    let (static_part, dynamic_part) = split_at_cache_boundary(&prompt);

    assert_eq!(static_part, "static part");
    assert_eq!(dynamic_part, "dynamic part");
}

#[test]
fn prompt_split_no_boundary() {
    let (s, d) = split_at_cache_boundary("no marker here");
    assert_eq!(s, "no marker here");
    assert_eq!(d, "");
}

#[test]
fn prompt_project_context() {
    let dir = temp_dir("project_ctx");

    // Create .cold.md
    std::fs::write(dir.join(".cold.md"), "# Project rules\nAlways test.").unwrap();

    let config = AgentConfig::new("test-model", 128_000, "sk-test")
        .with_root_dir(&dir);
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let prompt = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );

    assert!(prompt.contains("Project rules"));
    assert!(prompt.contains("Always test."));

    cleanup(&dir);
}

// ═══════════════════════════════════════════════════════════════
// 4. Skill Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn skill_load_from_dir() {
    let dir = temp_dir("skills");

    // Create a skill subdirectory with SKILL.md
    let skill_dir = dir.join("greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: greeting
description: A greeting skill
priority: 3
triggers:
  - "hello"
  - "hi"
---
When user says hello, greet them warmly.
"#,
    )
    .unwrap();

    let registry = SkillRegistry::load_from_dir(&dir).unwrap();
    let skills = registry.list();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "greeting");
    assert_eq!(skills[0].priority, 3);
    assert_eq!(skills[0].triggers, vec!["hello", "hi"]);

    cleanup(&dir);
}

#[test]
fn skill_trigger_matching() {
    let mut registry = SkillRegistry::new();
    registry.add(Skill {
        name: "rust-help".into(),
        description: "Rust assistance".into(),
        triggers: vec!["rust".into(), "cargo".into()],
        content: "Use `cargo build` for building.".into(),
        priority: 5,
    });

    // Match on "rust"
    let result = registry.build_prompt_injection("How do I compile Rust code?");
    assert!(result.is_some());
    assert!(result.unwrap().contains("cargo build"));

    // Match on "cargo" (case-insensitive)
    let result = registry.build_prompt_injection("Run CARGO test please");
    assert!(result.is_some());
}

#[test]
fn skill_priority_ordering() {
    let mut registry = SkillRegistry::new();

    registry.add(Skill {
        name: "low".into(),
        description: "low priority".into(),
        triggers: vec!["code".into()],
        content: "LOW_PRIORITY_CONTENT".into(),
        priority: 1,
    });
    registry.add(Skill {
        name: "high".into(),
        description: "high priority".into(),
        triggers: vec!["code".into()],
        content: "HIGH_PRIORITY_CONTENT".into(),
        priority: 10,
    });
    registry.add(Skill {
        name: "mid".into(),
        description: "mid priority".into(),
        triggers: vec!["code".into()],
        content: "MID_PRIORITY_CONTENT".into(),
        priority: 5,
    });

    let result = registry
        .build_prompt_injection("Help me write code")
        .unwrap();

    // High priority content should appear first
    let high_pos = result.find("HIGH_PRIORITY_CONTENT").unwrap();
    let mid_pos = result.find("MID_PRIORITY_CONTENT").unwrap();
    let low_pos = result.find("LOW_PRIORITY_CONTENT").unwrap();
    assert!(high_pos < mid_pos);
    assert!(mid_pos < low_pos);
}

#[test]
fn skill_no_match() {
    let mut registry = SkillRegistry::new();
    registry.add(Skill {
        name: "python".into(),
        description: "Python skills".into(),
        triggers: vec!["python".into(), "pip".into()],
        content: "Use pip install.".into(),
        priority: 1,
    });

    let result = registry.build_prompt_injection("How to compile Java?");
    assert!(result.is_none());
}

#[test]
fn skill_empty_message_returns_none() {
    let mut registry = SkillRegistry::new();
    registry.add(Skill {
        name: "test".into(),
        description: "test".into(),
        triggers: vec!["anything".into()],
        content: "content".into(),
        priority: 1,
    });

    assert!(registry.build_prompt_injection("").is_none());
}

// ═══════════════════════════════════════════════════════════════
// 5. Session Tests
// ═══════════════════════════════════════════════════════════════

#[tokio::test]
async fn session_save_and_load() {
    let dir = temp_dir("session_save");

    // Build a state with messages
    let mut state = ConversationState::new();
    state.add_message(ChatMessage::system("You are helpful"));
    state.add_message(ChatMessage::user("Hello"));
    state.add_message(ChatMessage::assistant("Hi!"));
    state.turn_count = 1;
    state.last_user_message = "Hello".to_string();

    // Save needs a compressor, build a minimal one
    let config = cold_context::CompressorConfig::new("test-model", 128_000);
    let client = cold_sdk::ColdClient::new("sk-fake").unwrap();
    let compressor = cold_context::ContextCompressor::new(config, client);

    cold_agent_sdk::session::save_session(&dir, &state, "test-model", &compressor)
        .await
        .unwrap();

    // Load it back
    let session_file = dir.join(format!("{}.json", state.session_id));
    let loaded = cold_agent_sdk::session::load_session(&session_file)
        .await
        .unwrap();

    assert_eq!(loaded.session_id, state.session_id);
    assert_eq!(loaded.model, "test-model");
    assert_eq!(loaded.turn_count, 1);
    assert_eq!(loaded.messages.len(), 3);

    cleanup(&dir);
}

#[tokio::test]
async fn session_metadata_auto_title() {
    let dir = temp_dir("session_title");

    let mut state = ConversationState::new();
    state.last_user_message = "Explain how async/await works in Rust".to_string();
    state.add_message(ChatMessage::user("Explain how async/await works in Rust"));

    let config = cold_context::CompressorConfig::new("test-model", 128_000);
    let client = cold_sdk::ColdClient::new("sk-fake").unwrap();
    let compressor = cold_context::ContextCompressor::new(config, client);

    cold_agent_sdk::session::save_session(&dir, &state, "test-model", &compressor)
        .await
        .unwrap();

    let session_file = dir.join(format!("{}.json", state.session_id));
    let loaded = cold_agent_sdk::session::load_session(&session_file)
        .await
        .unwrap();

    // Title should be auto-generated from first user message
    let meta = loaded.metadata.unwrap();
    assert!(meta.title.is_some());
    let title = meta.title.unwrap();
    assert!(title.contains("async/await"));

    cleanup(&dir);
}

#[tokio::test]
async fn session_list() {
    let dir = temp_dir("session_list");

    let config = cold_context::CompressorConfig::new("test-model", 128_000);
    let client = cold_sdk::ColdClient::new("sk-fake").unwrap();

    // Save 3 sessions
    for i in 0..3 {
        let mut state = ConversationState::new();
        state.last_user_message = format!("Session {i}");
        state.add_message(ChatMessage::user(format!("Session {i}")));

        let compressor = cold_context::ContextCompressor::new(config.clone(), client.clone());
        cold_agent_sdk::session::save_session(&dir, &state, "test-model", &compressor)
            .await
            .unwrap();
    }

    let sessions = list_sessions(&dir).await.unwrap();
    assert_eq!(sessions.len(), 3);

    cleanup(&dir);
}

#[tokio::test]
async fn session_jsonl_append() {
    let dir = temp_dir("session_jsonl");
    let session_id = uuid::Uuid::new_v4().to_string();

    // Append 3 messages
    append_to_log(&dir, &session_id, ChatMessage::system("sys"))
        .await
        .unwrap();
    append_to_log(&dir, &session_id, ChatMessage::user("usr"))
        .await
        .unwrap();
    append_to_log(&dir, &session_id, ChatMessage::assistant("ast"))
        .await
        .unwrap();

    // Load them back
    let messages = load_from_log(&dir, &session_id).await.unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, cold_sdk::Role::System);
    assert_eq!(messages[1].role, cold_sdk::Role::User);
    assert_eq!(messages[2].role, cold_sdk::Role::Assistant);

    cleanup(&dir);
}

// ═══════════════════════════════════════════════════════════════
// 6. Memory Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn memory_load_files() {
    let dir = temp_dir("memory");
    let memory_dir = dir.join(".cold").join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();

    std::fs::write(memory_dir.join("prefs.md"), "I prefer Rust.").unwrap();
    std::fs::write(memory_dir.join("context.md"), "Working on cold-agent.").unwrap();

    let entries = load_memory_files(&dir);
    assert_eq!(entries.len(), 2);

    // Sorted by name
    assert_eq!(entries[0].name, "context");
    assert!(entries[0].content.contains("cold-agent"));
    assert_eq!(entries[1].name, "prefs");
    assert!(entries[1].content.contains("Rust"));

    cleanup(&dir);
}

#[test]
fn memory_include_directive() {
    let dir = temp_dir("memory_include");
    let memory_dir = dir.join(".cold").join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();

    // Create an includable file
    std::fs::write(memory_dir.join("shared.md"), "Shared knowledge block.").unwrap();

    // Create a file that includes it
    std::fs::write(
        memory_dir.join("main.md"),
        "My preferences:\n@include:shared.md\nEnd.",
    )
    .unwrap();

    let entries = load_memory_files(&dir);
    // Find "main" entry
    let main_entry = entries.iter().find(|e| e.name == "main").unwrap();
    assert!(main_entry.content.contains("Shared knowledge block."));
    assert!(main_entry.content.contains("My preferences:"));
    assert!(main_entry.content.contains("End."));

    cleanup(&dir);
}

#[test]
fn memory_prompt_injection() {
    let entries = vec![
        MemoryEntry {
            name: "rules".into(),
            content: "Always write tests.".into(),
        },
    ];
    let prompt = build_memory_prompt(&entries);

    assert!(prompt.contains("# Memory"));
    assert!(prompt.contains("## rules"));
    assert!(prompt.contains("Always write tests."));
}

#[test]
fn memory_empty_prompt() {
    let prompt = build_memory_prompt(&[]);
    assert!(prompt.is_empty());
}

#[test]
fn memory_missing_dir() {
    let entries = load_memory_files(std::path::Path::new("/nonexistent/path/xyz"));
    assert!(entries.is_empty());
}

// ═══════════════════════════════════════════════════════════════
// 7. Hook Tests
// ═══════════════════════════════════════════════════════════════

/// A test hook that blocks specific tool calls and records events.
struct TestHook {
    blocked_tools: Vec<String>,
    events: Arc<Mutex<Vec<String>>>,
    block_stop: bool,
}

impl TestHook {
    fn new(blocked: Vec<&str>, block_stop: bool) -> Self {
        Self {
            blocked_tools: blocked.into_iter().map(String::from).collect(),
            events: Arc::new(Mutex::new(Vec::new())),
            block_stop,
        }
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl AgentHook for TestHook {
    fn is_active(&self) -> bool {
        true
    }

    fn on_session_start(&self, session_id: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("session_start:{session_id}"));
    }

    fn on_pre_tool_call(&self, name: &str, _args: &serde_json::Value) -> HookDecision {
        self.events
            .lock()
            .unwrap()
            .push(format!("pre_tool:{name}"));
        if self.blocked_tools.contains(&name.to_string()) {
            HookDecision::Block(format!("{name} is blocked by policy"))
        } else {
            HookDecision::Continue
        }
    }

    fn on_post_tool_call(&self, name: &str, _result: &cold_tools::ToolResult) {
        self.events
            .lock()
            .unwrap()
            .push(format!("post_tool:{name}"));
    }

    fn on_pre_compact(&self, count: usize) {
        self.events
            .lock()
            .unwrap()
            .push(format!("pre_compact:{count}"));
    }

    fn on_post_compact(&self, before: u32, after: u32) {
        self.events
            .lock()
            .unwrap()
            .push(format!("post_compact:{before}->{after}"));
    }

    fn on_turn_complete(&self, turn: u32) {
        self.events
            .lock()
            .unwrap()
            .push(format!("turn:{turn}"));
    }

    fn on_stop(&self) -> HookDecision {
        self.events.lock().unwrap().push("on_stop".to_string());
        if self.block_stop {
            HookDecision::Block("keep going".into())
        } else {
            HookDecision::Continue
        }
    }
}

#[test]
fn hook_pre_tool_call_block() {
    let hook = TestHook::new(vec!["dangerous_tool"], false);
    let args = serde_json::json!({"file": "/etc/passwd"});

    let decision = hook.on_pre_tool_call("dangerous_tool", &args);
    assert!(matches!(decision, HookDecision::Block(_)));

    if let HookDecision::Block(reason) = decision {
        assert!(reason.contains("dangerous_tool"));
        assert!(reason.contains("blocked"));
    }
}

#[test]
fn hook_pre_tool_call_continue() {
    let hook = TestHook::new(vec!["dangerous_tool"], false);
    let args = serde_json::json!({"query": "test"});

    let decision = hook.on_pre_tool_call("safe_tool", &args);
    assert!(matches!(decision, HookDecision::Continue));
}

#[test]
fn hook_on_stop_block() {
    let hook = TestHook::new(vec![], true);
    let decision = hook.on_stop();
    assert!(matches!(decision, HookDecision::Block(_)));
}

#[test]
fn hook_on_stop_continue() {
    let hook = TestHook::new(vec![], false);
    let decision = hook.on_stop();
    assert!(matches!(decision, HookDecision::Continue));
}

#[test]
fn hook_noop_is_inactive() {
    let noop = NoopHook;
    assert!(!noop.is_active());
    assert!(matches!(
        noop.on_pre_tool_call("any", &serde_json::json!({})),
        HookDecision::Continue
    ));
    assert!(matches!(noop.on_stop(), HookDecision::Continue));
}

#[test]
fn hook_events_recorded() {
    let hook = TestHook::new(vec!["blocked"], false);

    hook.on_session_start("sess-123");
    hook.on_pre_tool_call("safe", &serde_json::json!({}));
    hook.on_pre_tool_call("blocked", &serde_json::json!({}));
    hook.on_turn_complete(1);
    hook.on_stop();

    let events = hook.events();
    assert_eq!(events.len(), 5);
    assert_eq!(events[0], "session_start:sess-123");
    assert_eq!(events[1], "pre_tool:safe");
    assert_eq!(events[2], "pre_tool:blocked");
    assert_eq!(events[3], "turn:1");
    assert_eq!(events[4], "on_stop");
}

// ═══════════════════════════════════════════════════════════════
// 8. Callback Tests
// ═══════════════════════════════════════════════════════════════

/// A test callback that records all events.
struct RecordingCallback {
    events: Arc<Mutex<Vec<String>>>,
}

impl RecordingCallback {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

impl AgentCallback for RecordingCallback {
    fn on_text(&self, text: &str) {
        self.events
            .lock()
            .unwrap()
            .push(format!("text:{text}"));
    }

    fn on_tool_call(&self, name: &str, _args: &serde_json::Value) {
        self.events
            .lock()
            .unwrap()
            .push(format!("tool_call:{name}"));
    }

    fn on_tool_result(&self, name: &str, _result: &cold_tools::ToolResult) {
        self.events
            .lock()
            .unwrap()
            .push(format!("tool_result:{name}"));
    }

    fn on_error(&self, error: &cold_agent_sdk::AgentError) {
        self.events
            .lock()
            .unwrap()
            .push(format!("error:{error}"));
    }

    fn on_complete(&self, result: &cold_agent_sdk::AgentResult) {
        self.events
            .lock()
            .unwrap()
            .push(format!("complete:turns={}", result.turns_used));
    }

    fn on_compress(&self, before: u32, after: u32) {
        self.events
            .lock()
            .unwrap()
            .push(format!("compress:{before}->{after}"));
    }

    fn on_progress(&self, turn: u32, max_turns: u32) {
        self.events
            .lock()
            .unwrap()
            .push(format!("progress:{turn}/{max_turns}"));
    }
}

#[test]
fn callback_on_text_fires() {
    let cb = RecordingCallback::new();
    cb.on_text("Hello world");
    cb.on_text(" more text");

    let events = cb.events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0], "text:Hello world");
    assert_eq!(events[1], "text: more text");
}

#[test]
fn callback_on_tool_call_fires() {
    let cb = RecordingCallback::new();
    let args = serde_json::json!({"path": "/src/main.rs"});
    cb.on_tool_call("read_file", &args);

    let events = cb.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], "tool_call:read_file");
}

#[test]
fn callback_on_progress_fires() {
    let cb = RecordingCallback::new();
    cb.on_progress(5, 90);

    let events = cb.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], "progress:5/90");
}

#[test]
fn callback_on_complete_fires() {
    let cb = RecordingCallback::new();
    let result = AgentResult {
        text: "Done".into(),
        turns_used: 3,
        tokens: TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 200,
            total_tokens: 1200,
        },
        tools_called: vec![],
        compressed: false,
    };
    cb.on_complete(&result);

    let events = cb.events();
    assert_eq!(events[0], "complete:turns=3");
}

#[test]
fn callback_silent_does_nothing() {
    // SilentCallback should not panic on any event
    let cb = SilentCallback;
    cb.on_text("test");
    cb.on_tool_call("tool", &serde_json::json!({}));
    cb.on_progress(1, 10);
    cb.on_compress(5000, 1000);
}

// ═══════════════════════════════════════════════════════════════
// 9. Config Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn config_defaults() {
    let config = AgentConfig::new("claude-sonnet-4-20250514", 200_000, "sk-key");

    assert_eq!(config.model, "claude-sonnet-4-20250514");
    assert_eq!(config.context_length, 200_000);
    assert_eq!(config.max_turns, 90);
    assert!(config.streaming);
    assert!(config.auto_compress);
    assert!(config.persist_session);
    assert!(config.base_url.is_none());
    assert!(config.system_prompt.is_none());
    assert!(config.skills_dir.is_none());
    assert!(config.session_dir.is_none());
    assert!(config.fallback_model.is_none());
    assert!(config.fallback_base_url.is_none());
    assert_eq!(config.permission_mode, cold_tools::PermissionMode::Default);
}

#[test]
fn config_builder() {
    let config = AgentConfig::new("gpt-4o", 128_000, "sk-key")
        .with_base_url("https://api.example.com")
        .with_root_dir("/my/project")
        .with_system_prompt("Custom prompt")
        .with_max_turns(50)
        .with_streaming(false)
        .with_auto_compress(false)
        .with_persist_session(false)
        .with_session_dir("/tmp/sessions")
        .with_skills_dir("/skills")
        .with_permission_mode(cold_tools::PermissionMode::Plan)
        .with_fallback_model("gpt-4o-mini")
        .with_fallback_base_url("https://fallback.example.com");

    assert_eq!(config.model, "gpt-4o");
    assert_eq!(config.base_url.as_deref(), Some("https://api.example.com"));
    assert_eq!(config.root_dir, PathBuf::from("/my/project"));
    assert_eq!(config.system_prompt.as_deref(), Some("Custom prompt"));
    assert_eq!(config.max_turns, 50);
    assert!(!config.streaming);
    assert!(!config.auto_compress);
    assert!(!config.persist_session);
    assert_eq!(config.session_dir, Some(PathBuf::from("/tmp/sessions")));
    assert_eq!(config.skills_dir, Some(PathBuf::from("/skills")));
    assert_eq!(config.permission_mode, cold_tools::PermissionMode::Plan);
    assert_eq!(config.fallback_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(
        config.fallback_base_url.as_deref(),
        Some("https://fallback.example.com")
    );
}

#[test]
fn config_debug_redacts_key() {
    let config = AgentConfig::new("model", 128_000, "sk-super-secret-key-12345");
    let debug = format!("{config:?}");

    assert!(debug.contains("[REDACTED]"));
    assert!(!debug.contains("sk-super-secret-key-12345"));
}

// ═══════════════════════════════════════════════════════════════
// 10. Streaming Executor Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn executor_feed_text() {
    let mut executor = StreamingToolExecutor::new();

    executor.feed_text("Hello ");
    executor.feed_text("world");
    executor.feed_text("!");

    assert_eq!(executor.text(), "Hello world!");
}

#[test]
fn executor_take_text() {
    let mut executor = StreamingToolExecutor::new();
    executor.feed_text("some content");

    let text = executor.take_text();
    assert_eq!(text, "some content");

    // After take, text is empty
    assert_eq!(executor.text(), "");
}

#[test]
fn executor_feed_tool_calls() {
    let mut executor = StreamingToolExecutor::new();

    // First chunk: tool call start with id + name
    executor.feed_tool_calls(&[StreamToolCall {
        index: 0,
        id: Some("call_abc".into()),
        call_type: Some("function".into()),
        function: Some(StreamFunctionCall {
            name: Some("read_file".into()),
            arguments: Some(r#"{"pa"#.into()),
        }),
    }]);

    // Second chunk: more arguments
    executor.feed_tool_calls(&[StreamToolCall {
        index: 0,
        id: None,
        call_type: None,
        function: Some(StreamFunctionCall {
            name: None,
            arguments: Some(r#"th": "/src"}"#.into()),
        }),
    }]);

    // Take the assembled tool calls
    let tcs = executor.take_tool_calls();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].id, "call_abc");
    assert_eq!(tcs[0].function.name, "read_file");
    assert_eq!(tcs[0].function.arguments, r#"{"path": "/src"}"#);
}

#[test]
fn executor_multiple_tool_calls() {
    let mut executor = StreamingToolExecutor::new();

    // Two tool calls in parallel
    executor.feed_tool_calls(&[
        StreamToolCall {
            index: 0,
            id: Some("call_1".into()),
            call_type: Some("function".into()),
            function: Some(StreamFunctionCall {
                name: Some("tool_a".into()),
                arguments: Some(r#"{"x": 1}"#.into()),
            }),
        },
        StreamToolCall {
            index: 1,
            id: Some("call_2".into()),
            call_type: Some("function".into()),
            function: Some(StreamFunctionCall {
                name: Some("tool_b".into()),
                arguments: Some(r#"{"y": 2}"#.into()),
            }),
        },
    ]);

    let tcs = executor.take_tool_calls();
    assert_eq!(tcs.len(), 2);
    assert_eq!(tcs[0].function.name, "tool_a");
    assert_eq!(tcs[1].function.name, "tool_b");
}

#[test]
fn executor_validate_args_valid() {
    let mut executor = StreamingToolExecutor::new();

    executor.feed_tool_calls(&[StreamToolCall {
        index: 0,
        id: Some("call_1".into()),
        call_type: Some("function".into()),
        function: Some(StreamFunctionCall {
            name: Some("tool".into()),
            arguments: Some(r#"{"key": "value"}"#.into()),
        }),
    }]);

    let invalid = executor.validate_all_args();
    assert!(invalid.is_empty());
}

#[test]
fn executor_validate_args_invalid() {
    let mut executor = StreamingToolExecutor::new();

    executor.feed_tool_calls(&[StreamToolCall {
        index: 0,
        id: Some("call_1".into()),
        call_type: Some("function".into()),
        function: Some(StreamFunctionCall {
            name: Some("tool".into()),
            arguments: Some(r#"{"key": broken"#.into()),
        }),
    }]);

    let invalid = executor.validate_all_args();
    assert_eq!(invalid, vec![0]);
}

#[test]
fn executor_finish_reason() {
    let mut executor = StreamingToolExecutor::new();

    assert!(executor.finish_reason().is_none());

    executor.set_finish_reason(FinishReason::Stop);
    assert_eq!(executor.finish_reason(), Some(&FinishReason::Stop));

    // Tool calls finish reason
    let mut executor2 = StreamingToolExecutor::new();
    executor2.set_finish_reason(FinishReason::ToolCalls);
    assert_eq!(executor2.finish_reason(), Some(&FinishReason::ToolCalls));
}

#[test]
fn executor_has_tool_calls() {
    let mut executor = StreamingToolExecutor::new();
    assert!(!executor.has_tool_calls());

    executor.feed_tool_calls(&[StreamToolCall {
        index: 0,
        id: Some("call_1".into()),
        call_type: None,
        function: Some(StreamFunctionCall {
            name: Some("tool".into()),
            arguments: Some("{}".into()),
        }),
    }]);

    assert!(executor.has_tool_calls());
}

#[test]
fn executor_feed_and_check_ready() {
    let mut executor = StreamingToolExecutor::new();

    // First chunk: partial arguments (not valid JSON yet)
    let ready1 = executor.feed_and_check(&[StreamToolCall {
        index: 0,
        id: Some("call_1".into()),
        call_type: Some("function".into()),
        function: Some(StreamFunctionCall {
            name: Some("my_tool".into()),
            arguments: Some(r#"{"ke"#.into()),
        }),
    }]);
    assert!(ready1.is_empty()); // Not ready yet

    // Second chunk: completes the JSON
    let ready2 = executor.feed_and_check(&[StreamToolCall {
        index: 0,
        id: None,
        call_type: None,
        function: Some(StreamFunctionCall {
            name: None,
            arguments: Some(r#"y": 42}"#.into()),
        }),
    }]);
    assert_eq!(ready2.len(), 1);
    assert_eq!(ready2[0].tool_call.function.name, "my_tool");
    assert_eq!(ready2[0].tool_call.function.arguments, r#"{"key": 42}"#);
}

// ═══════════════════════════════════════════════════════════════
// 11. Error Display Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn error_budget_exhausted_display() {
    let err = AgentError::BudgetExhausted {
        turns_used: 90,
        max_turns: 90,
    };
    let msg = format!("{err}");
    assert!(msg.contains("90"));
    assert!(msg.contains("budget exhausted"));
}

#[test]
fn error_interrupted_display() {
    let err = AgentError::Interrupted;
    let msg = format!("{err}");
    assert!(msg.contains("interrupted"));
}

#[test]
fn error_config_display() {
    let err = AgentError::Config("missing field".into());
    let msg = format!("{err}");
    assert!(msg.contains("missing field"));
}

// ═══════════════════════════════════════════════════════════════
// 12. AgentResult Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn agent_result_fields() {
    let result = AgentResult {
        text: "The answer is 42.".into(),
        turns_used: 5,
        tokens: TokenUsage {
            prompt_tokens: 2000,
            completion_tokens: 500,
            total_tokens: 2500,
        },
        tools_called: vec![ToolCallRecord {
            name: "calculator".into(),
            args: serde_json::json!({"expr": "6*7"}),
            result_preview: "42".into(),
            duration_ms: 10,
            succeeded: true,
        }],
        compressed: true,
    };

    assert_eq!(result.text, "The answer is 42.");
    assert_eq!(result.turns_used, 5);
    assert_eq!(result.tokens.total_tokens, 2500);
    assert_eq!(result.tools_called.len(), 1);
    assert_eq!(result.tools_called[0].name, "calculator");
    assert!(result.tools_called[0].succeeded);
    assert!(result.compressed);
}

// ═══════════════════════════════════════════════════════════════
// 13. Session SavedSession Serde Round-trip
// ═══════════════════════════════════════════════════════════════

#[test]
fn saved_session_serde_roundtrip() {
    let session = SavedSession {
        session_id: "test-id-123".into(),
        created_at: "1700000000".into(),
        model: "gpt-4o".into(),
        messages: vec![
            serde_json::to_value(ChatMessage::user("hello")).unwrap(),
            serde_json::to_value(ChatMessage::assistant("hi")).unwrap(),
        ],
        compressor_state: cold_context::CompressorState {
            last_prompt_tokens: 100,
            last_completion_tokens: 50,
            compression_count: 0,
            previous_summary: None,
            ineffective_count: 0,
            last_savings_pct: 0.0,
        },
        turn_count: 1,
        metadata: Some(SessionMetadata {
            session_id: "test-id-123".into(),
            title: Some("hello".into()),
            tags: vec!["test".into()],
            created_at: "1700000000".into(),
            updated_at: "1700000000".into(),
            model: "gpt-4o".into(),
            turn_count: 1,
            total_tokens: 150,
        }),
    };

    let json = serde_json::to_string(&session).unwrap();
    let deserialized: SavedSession = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.session_id, "test-id-123");
    assert_eq!(deserialized.messages.len(), 2);
    assert_eq!(deserialized.turn_count, 1);
    assert!(deserialized.metadata.is_some());
}

// ═══════════════════════════════════════════════════════════════
// 14. Skill File Loading from Disk
// ═══════════════════════════════════════════════════════════════

#[test]
fn skill_load_recursive() {
    let dir = temp_dir("skill_recursive");

    // Create nested skill directories
    let a_dir = dir.join("a");
    let b_dir = dir.join("a").join("b");
    std::fs::create_dir_all(&b_dir).unwrap();

    std::fs::write(
        a_dir.join("SKILL.md"),
        "---\nname: skill-a\ndescription: A\npriority: 1\ntriggers:\n  - \"alpha\"\n---\nContent A",
    )
    .unwrap();

    std::fs::write(
        b_dir.join("SKILL.md"),
        "---\nname: skill-b\ndescription: B\npriority: 2\ntriggers:\n  - \"beta\"\n---\nContent B",
    )
    .unwrap();

    let registry = SkillRegistry::load_from_dir(&dir).unwrap();
    let skills = registry.list();

    assert_eq!(skills.len(), 2);
    let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"skill-a"));
    assert!(names.contains(&"skill-b"));

    cleanup(&dir);
}

#[test]
fn skill_load_nonexistent_dir() {
    // Should return empty registry, not error
    let registry = SkillRegistry::load_from_dir(std::path::Path::new("/nonexistent/dir/xyz"))
        .unwrap();
    assert!(registry.list().is_empty());
}

// ═══════════════════════════════════════════════════════════════
// 15. Budget Edge Cases
// ═══════════════════════════════════════════════════════════════

#[test]
fn budget_zero_turns() {
    let mut budget = IterationBudget::new(0);

    // No normal turns, but grace is still available
    assert_eq!(budget.remaining(), 0);
    assert!(budget.has_remaining()); // grace
    assert!(budget.consume()); // grace turn
    assert!(!budget.has_remaining());
    assert!(!budget.consume());
}

#[test]
fn budget_reset_restores_grace() {
    let mut budget = IterationBudget::new(1);
    budget.consume(); // normal
    budget.consume(); // grace
    assert!(!budget.has_remaining());

    budget.reset();
    assert!(budget.has_remaining());
    assert_eq!(budget.used(), 0);
    assert_eq!(budget.remaining(), 1);

    // Grace is also restored
    budget.consume(); // normal
    assert!(budget.has_remaining()); // grace available again
    budget.consume(); // grace
    assert!(!budget.has_remaining());
}

// ═══════════════════════════════════════════════════════════════
// 16. Memory Integration with System Prompt
// ═══════════════════════════════════════════════════════════════

#[test]
fn memory_in_system_prompt() {
    let dir = temp_dir("mem_prompt");
    let memory_dir = dir.join(".cold").join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();
    std::fs::write(memory_dir.join("notes.md"), "Remember: use Arc<Mutex>.").unwrap();

    let config = AgentConfig::new("test-model", 128_000, "sk-test")
        .with_root_dir(&dir);
    let skills = SkillRegistry::new();
    let state = ConversationState::new();

    let prompt = cold_agent_sdk::prompt::build_system_prompt(
        &config, &skills, &state, false, false,
    );

    assert!(prompt.contains("# Memory"));
    assert!(prompt.contains("Remember: use Arc<Mutex>."));

    cleanup(&dir);
}
