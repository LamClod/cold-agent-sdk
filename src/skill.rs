use std::path::Path;

use crate::error::AgentError;

/// A loaded skill with metadata parsed from YAML frontmatter.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Human-readable skill name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Trigger keywords (matched case-insensitively against user input).
    pub triggers: Vec<String>,
    /// The skill body content (markdown after the frontmatter).
    pub content: String,
    /// Priority (higher = injected first).
    pub priority: u8,
}

/// Registry of skills loaded from disk.
#[derive(Debug, Clone)]
pub struct SkillRegistry {
    skills: Vec<Skill>,
}

impl SkillRegistry {
    /// Create an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self { skills: Vec::new() }
    }

    /// Scan a directory for `SKILL.md` files and load them.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::SessionIo` on filesystem errors.
    pub fn load_from_dir(dir: &Path) -> Result<Self, AgentError> {
        let mut registry = Self::new();
        if !dir.is_dir() {
            return Ok(registry);
        }
        load_skills_recursive(dir, &mut registry)?;
        Ok(registry)
    }

    /// Check each skill's triggers against `user_message` and return the
    /// combined content of all matching skills, sorted by priority (highest
    /// first).
    #[must_use]
    pub fn build_prompt_injection(&self, user_message: &str) -> Option<String> {
        if user_message.is_empty() {
            return None;
        }
        let lower = user_message.to_lowercase();
        let mut matched: Vec<&Skill> = self
            .skills
            .iter()
            .filter(|s| {
                s.triggers
                    .iter()
                    .any(|t| lower.contains(&t.to_lowercase()))
            })
            .collect();

        if matched.is_empty() {
            return None;
        }

        matched.sort_by_key(|s| std::cmp::Reverse(s.priority));
        let combined: Vec<&str> = matched.iter().map(|s| s.content.as_str()).collect();
        Some(combined.join("\n\n---\n\n"))
    }

    /// Add a skill to the registry.
    pub fn add(&mut self, skill: Skill) {
        self.skills.push(skill);
    }

    /// List all loaded skills.
    #[must_use]
    pub fn list(&self) -> &[Skill] {
        &self.skills
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively walk a directory looking for `SKILL.md` files.
fn load_skills_recursive(dir: &Path, registry: &mut SkillRegistry) -> Result<(), AgentError> {
    let entries = std::fs::read_dir(dir).map_err(AgentError::SessionIo)?;
    for entry in entries {
        let entry = entry.map_err(AgentError::SessionIo)?;
        let path = entry.path();
        if path.is_dir() {
            load_skills_recursive(&path, registry)?;
        } else if path
            .file_name()
            .is_some_and(|n| n.eq_ignore_ascii_case("SKILL.md"))
        {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Some(skill) = parse_skill_file(&content) {
                    registry.add(skill);
                }
            }
        }
    }
    Ok(())
}

/// Parse a SKILL.md file with YAML frontmatter delimited by `---`.
fn parse_skill_file(raw: &str) -> Option<Skill> {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }
    // Find the closing ---
    let after_open = &trimmed[3..];
    let close_idx = after_open.find("\n---")?;
    let frontmatter = &after_open[..close_idx];
    let body = after_open[close_idx + 4..].trim_start().to_string();

    // Minimal YAML parsing (no dependency on a yaml crate)
    let mut name = String::new();
    let mut description = String::new();
    let mut triggers: Vec<String> = Vec::new();
    let mut priority: u8 = 0;

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("name:") {
            name = val.trim().trim_matches('"').to_string();
        } else if let Some(val) = line.strip_prefix("description:") {
            description = val.trim().trim_matches('"').to_string();
        } else if let Some(val) = line.strip_prefix("priority:") {
            priority = val.trim().parse().unwrap_or(0);
        } else if line.starts_with("triggers:") {
            // triggers will be on subsequent lines starting with " - "
        } else if let Some(val) = line.strip_prefix("- ") {
            triggers.push(val.trim().trim_matches('"').to_string());
        }
    }

    if name.is_empty() {
        return None;
    }

    Some(Skill {
        name,
        description,
        triggers,
        content: body,
        priority,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_skill_file() {
        let raw = r#"---
name: test-skill
description: A test skill
priority: 5
triggers:
  - "hello"
  - "world"
---
This is the skill content.
"#;
        let skill = parse_skill_file(raw).unwrap();
        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "A test skill");
        assert_eq!(skill.priority, 5);
        assert_eq!(skill.triggers, vec!["hello", "world"]);
        assert!(skill.content.contains("This is the skill content."));
    }

    #[test]
    fn test_build_prompt_injection_match() {
        let mut registry = SkillRegistry::new();
        registry.add(Skill {
            name: "greet".into(),
            description: "greeting".into(),
            triggers: vec!["hello".into()],
            content: "Say hi".into(),
            priority: 1,
        });
        let injection = registry.build_prompt_injection("Hello there");
        assert_eq!(injection, Some("Say hi".into()));
    }

    #[test]
    fn test_build_prompt_injection_no_match() {
        let mut registry = SkillRegistry::new();
        registry.add(Skill {
            name: "greet".into(),
            description: "greeting".into(),
            triggers: vec!["hello".into()],
            content: "Say hi".into(),
            priority: 1,
        });
        let injection = registry.build_prompt_injection("goodbye");
        assert!(injection.is_none());
    }
}
