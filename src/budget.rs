/// Tracks how many agentic loop iterations remain.
///
/// A single grace turn is granted after the hard limit to allow the model
/// to finish gracefully.
#[derive(Debug, Clone)]
pub struct IterationBudget {
    max: u32,
    used: u32,
    grace_remaining: u32,
}

impl IterationBudget {
    /// Create a new budget with the given maximum turn count.
    #[must_use]
    pub const fn new(max_turns: u32) -> Self {
        Self {
            max: max_turns,
            used: 0,
            grace_remaining: 1,
        }
    }

    /// Consume one turn. Returns `true` if the turn was granted (including
    /// a single grace turn), `false` when fully exhausted.
    pub const fn consume(&mut self) -> bool {
        if self.used < self.max {
            self.used += 1;
            true
        } else if self.grace_remaining > 0 {
            self.grace_remaining -= 1;
            self.used += 1;
            true
        } else {
            false
        }
    }

    /// Whether there are any turns left (including grace).
    #[must_use]
    pub const fn has_remaining(&self) -> bool {
        self.used < self.max || self.grace_remaining > 0
    }

    /// How many normal (non-grace) turns remain.
    #[must_use]
    pub const fn remaining(&self) -> u32 {
        self.max.saturating_sub(self.used)
    }

    /// How many turns have been consumed so far.
    #[must_use]
    pub const fn used(&self) -> u32 {
        self.used
    }

    /// Give back one turn (e.g. when a turn did not actually call the model).
    pub const fn refund(&mut self) {
        if self.used > 0 {
            self.used -= 1;
        }
    }

    /// Reset to the initial state.
    pub const fn reset(&mut self) {
        self.used = 0;
        self.grace_remaining = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_budget() {
        let mut b = IterationBudget::new(3);
        assert!(b.has_remaining());
        assert_eq!(b.remaining(), 3);

        assert!(b.consume());
        assert!(b.consume());
        assert!(b.consume());
        // Grace turn
        assert!(b.has_remaining());
        assert!(b.consume());
        // Now fully exhausted
        assert!(!b.has_remaining());
        assert!(!b.consume());
    }

    #[test]
    fn test_refund() {
        let mut b = IterationBudget::new(2);
        b.consume();
        b.consume();
        assert_eq!(b.remaining(), 0);
        b.refund();
        assert_eq!(b.remaining(), 1);
    }

    #[test]
    fn test_reset() {
        let mut b = IterationBudget::new(2);
        b.consume();
        b.consume();
        b.consume(); // grace
        b.reset();
        assert_eq!(b.used(), 0);
        assert!(b.has_remaining());
    }
}
