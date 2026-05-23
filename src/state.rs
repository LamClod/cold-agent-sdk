use cold_sdk::ChatMessage;

/// Mutable conversation state tracked across the agent loop.
pub struct ConversationState {
    /// Full message history sent to the model.
    pub messages: Vec<ChatMessage>,
    /// Unique identifier for this session.
    pub session_id: String,
    /// Number of agentic turns completed.
    pub turn_count: u32,
    /// Running total of prompt tokens.
    pub total_prompt_tokens: u32,
    /// Running total of completion tokens.
    pub total_completion_tokens: u32,
    /// A note injected into the system prompt after compression.
    pub compression_note: Option<String>,
    /// The most recent user message (used for skill matching).
    pub last_user_message: String,
}

impl ConversationState {
    /// Create a fresh state with a random session id.
    #[must_use]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            session_id: uuid::Uuid::new_v4().to_string(),
            turn_count: 0,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            compression_note: None,
            last_user_message: String::new(),
        }
    }

    /// Append a message to the history.
    pub fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
    }

    /// Append a tool-result message.
    pub fn add_tool_result(&mut self, tool_call_id: &str, content: &str) {
        self.messages
            .push(ChatMessage::tool(tool_call_id, content));
    }

    /// Set the compression note that will be included in the next system prompt.
    pub fn set_compression_note(&mut self, note: String) {
        self.compression_note = Some(note);
    }

    /// Take all messages out, leaving an empty vec in place.
    pub fn take_messages(&mut self) -> Vec<ChatMessage> {
        std::mem::take(&mut self.messages)
    }

    /// Replace the message history wholesale.
    pub fn restore_messages(&mut self, messages: Vec<ChatMessage>) {
        self.messages = messages;
    }
}

impl Default for ConversationState {
    fn default() -> Self {
        Self::new()
    }
}
