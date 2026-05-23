use std::path::Path;

use serde::{Deserialize, Serialize};

use cold_context::CompressorState;
use cold_sdk::ChatMessage;

use crate::error::AgentError;
use crate::state::ConversationState;

/// A serializable snapshot of a conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedSession {
    /// Unique session identifier.
    pub session_id: String,
    /// ISO-ish creation timestamp.
    pub created_at: String,
    /// Model used for the session.
    pub model: String,
    /// Serialized message history.
    pub messages: Vec<serde_json::Value>,
    /// Compressor runtime state snapshot.
    pub compressor_state: CompressorState,
    /// Number of agentic turns completed.
    pub turn_count: u32,
    /// Optional session metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<SessionMetadata>,
}

/// Metadata about a session for indexing and display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Unique session identifier.
    pub session_id: String,
    /// Auto-generated title from the first user message.
    pub title: Option<String>,
    /// User-defined tags for categorisation.
    pub tags: Vec<String>,
    /// Unix timestamp when the session was created.
    pub created_at: String,
    /// Unix timestamp when the session was last updated.
    pub updated_at: String,
    /// Model used for the session.
    pub model: String,
    /// Number of agentic turns completed.
    pub turn_count: u32,
    /// Total tokens consumed (prompt + completion).
    pub total_tokens: u32,
}

/// Persist the current conversation state to disk.
///
/// # Errors
///
/// Returns `AgentError::SessionIo` on filesystem errors, or
/// `AgentError::Config` if serialization fails.
pub async fn save_session<S: cold_context::Summarizer>(
    dir: &Path,
    state: &ConversationState,
    model: &str,
    compressor: &cold_context::ContextCompressor<S>,
) -> Result<(), AgentError> {
    tokio::fs::create_dir_all(dir)
        .await
        .map_err(AgentError::SessionIo)?;

    let messages: Vec<serde_json::Value> = state
        .messages
        .iter()
        .filter_map(|m| serde_json::to_value(m).ok())
        .collect();

    let now = std::time::SystemTime::now();
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let title = generate_title(state);
    let metadata = SessionMetadata {
        session_id: state.session_id.clone(),
        title: Some(title),
        tags: Vec::new(),
        created_at: format!("{secs}"),
        updated_at: format!("{secs}"),
        model: model.to_string(),
        turn_count: state.turn_count,
        total_tokens: state.total_prompt_tokens + state.total_completion_tokens,
    };

    let session = SavedSession {
        session_id: state.session_id.clone(),
        created_at: format!("{secs}"),
        model: model.to_string(),
        messages,
        compressor_state: compressor.save_state(),
        turn_count: state.turn_count,
        metadata: Some(metadata),
    };

    let json =
        serde_json::to_string_pretty(&session).map_err(|e| AgentError::Config(e.to_string()))?;

    let path = dir.join(format!("{}.json", state.session_id));
    tokio::fs::write(&path, json)
        .await
        .map_err(AgentError::SessionIo)?;

    Ok(())
}

/// Load a previously saved session from a JSON file.
///
/// # Errors
///
/// Returns `AgentError::SessionIo` on read failure, or `AgentError::Config`
/// on deserialization failure.
pub async fn load_session(path: &Path) -> Result<SavedSession, AgentError> {
    let data = tokio::fs::read_to_string(path)
        .await
        .map_err(AgentError::SessionIo)?;

    serde_json::from_str(&data).map_err(|e| AgentError::Config(e.to_string()))
}

// ─── JSONL Append Log ────────────────────────────────────────

/// Append a single message to the JSONL transcript log.
///
/// Each line in the JSONL file is a self-contained JSON serialization of one
/// `ChatMessage`. This provides crash-recoverable, append-only logging that
/// does not require rewriting the entire session file on every message.
///
/// File path: `{session_dir}/{session_id}.jsonl`
///
/// # Errors
///
/// Returns `AgentError::SessionIo` on filesystem errors, or
/// `AgentError::Config` if serialization fails.
pub async fn append_to_log(
    session_dir: &Path,
    session_id: &str,
    message: ChatMessage,
) -> Result<(), AgentError> {
    use tokio::io::AsyncWriteExt;

    tokio::fs::create_dir_all(session_dir)
        .await
        .map_err(AgentError::SessionIo)?;

    let line =
        serde_json::to_string(&message).map_err(|e| AgentError::Config(e.to_string()))?;

    let path = session_dir.join(format!("{session_id}.jsonl"));
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(AgentError::SessionIo)?;

    file.write_all(line.as_bytes())
        .await
        .map_err(AgentError::SessionIo)?;
    file.write_all(b"\n")
        .await
        .map_err(AgentError::SessionIo)?;

    Ok(())
}

/// Save session metadata to a separate `.meta.json` file.
///
/// # Errors
///
/// Returns `AgentError::SessionIo` on filesystem errors, or
/// `AgentError::Config` if serialization fails.
pub async fn save_metadata(
    dir: &Path,
    metadata: &SessionMetadata,
) -> Result<(), AgentError> {
    tokio::fs::create_dir_all(dir)
        .await
        .map_err(AgentError::SessionIo)?;

    let json = serde_json::to_string_pretty(metadata)
        .map_err(|e| AgentError::Config(e.to_string()))?;

    let path = dir.join(format!("{}.meta.json", metadata.session_id));
    tokio::fs::write(&path, json)
        .await
        .map_err(AgentError::SessionIo)?;

    Ok(())
}

/// Load session metadata from a `.meta.json` file.
///
/// # Errors
///
/// Returns `AgentError::SessionIo` on read failure, or `AgentError::Config`
/// on deserialization failure.
pub async fn load_metadata(path: &Path) -> Result<SessionMetadata, AgentError> {
    let data = tokio::fs::read_to_string(path)
        .await
        .map_err(AgentError::SessionIo)?;

    serde_json::from_str(&data).map_err(|e| AgentError::Config(e.to_string()))
}

/// List all sessions in a directory with their metadata.
///
/// Scans for `*.json` files (excluding `.meta.json` and `.jsonl`) and
/// attempts to load each as a `SavedSession`. Returns metadata for each
/// successfully loaded session.
///
/// # Errors
///
/// Returns `AgentError::SessionIo` if the directory cannot be read.
pub async fn list_sessions(dir: &Path) -> Result<Vec<SessionMetadata>, AgentError> {
    let mut entries = tokio::fs::read_dir(dir)
        .await
        .map_err(AgentError::SessionIo)?;

    let mut results: Vec<SessionMetadata> = Vec::new();

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        // Skip non-session files.
        let path_ref = std::path::Path::new(name);
        let is_json = path_ref
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        let is_meta = name.to_ascii_lowercase().ends_with(".meta.json");
        let is_jsonl = path_ref
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"));
        if !is_json || is_meta || is_jsonl {
            continue;
        }

        // Try loading full session first, extract metadata.
        if let Ok(session) = load_session(&path).await {
            if let Some(meta) = session.metadata {
                results.push(meta);
            } else {
                // Build minimal metadata from the session itself.
                results.push(SessionMetadata {
                    session_id: session.session_id,
                    title: None,
                    tags: Vec::new(),
                    created_at: session.created_at.clone(),
                    updated_at: session.created_at,
                    model: session.model,
                    turn_count: session.turn_count,
                    total_tokens: 0,
                });
            }
        }
    }

    // Sort by created_at descending (newest first).
    results.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(results)
}

/// Generate a title from the first user message (first 50 chars).
fn generate_title(state: &ConversationState) -> String {
    let msg = if state.last_user_message.is_empty() {
        // Fall back to finding the first user message in history.
        state
            .messages
            .iter()
            .find(|m| m.role == cold_sdk::Role::User)
            .and_then(|m| match &m.content {
                Some(cold_sdk::MessageContent::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .unwrap_or("Untitled session")
    } else {
        &state.last_user_message
    };

    // Take first 50 chars, trim to word boundary if possible.
    if msg.len() <= 50 {
        return msg.to_string();
    }
    let truncated = &msg[..50];
    if let Some(last_space) = truncated.rfind(' ') {
        if last_space > 20 {
            return format!("{}...", &truncated[..last_space]);
        }
    }
    format!("{truncated}...")
}

/// Read a session transcript from a JSONL log file (for crash recovery).
///
/// Each line is parsed as a `ChatMessage`. Malformed lines are silently
/// skipped to maximise recovery.
///
/// # Errors
///
/// Returns `AgentError::SessionIo` if the file cannot be read.
pub async fn load_from_log(
    session_dir: &Path,
    session_id: &str,
) -> Result<Vec<ChatMessage>, AgentError> {
    let path = session_dir.join(format!("{session_id}.jsonl"));
    let data = tokio::fs::read_to_string(&path)
        .await
        .map_err(AgentError::SessionIo)?;

    let messages = data
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<ChatMessage>(line).ok())
        .collect();

    Ok(messages)
}
