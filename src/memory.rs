/// Memory system for persistent context injection.
///
/// Memory entries are loaded from `.cold/memory/*.md` files in the project
/// root. Each file becomes a named memory entry whose content is injected
/// into the system prompt.
use std::path::Path;

/// A single memory entry loaded from disk.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Name derived from the filename (without `.md`).
    pub name: String,
    /// Full markdown content of the memory file.
    pub content: String,
}

/// Maximum bytes to read from a single memory file.
const MAX_MEMORY_FILE_BYTES: u64 = 30 * 1024;

/// Load all memory entries from `{root}/.cold/memory/`.
///
/// Each `.md` file in the directory becomes a [`MemoryEntry`]. Files larger
/// than 30 KB are truncated. The `@include:filename.md` directive is
/// resolved to inline the referenced file from the same directory.
#[must_use]
pub fn load_memory_files(root: &Path) -> Vec<MemoryEntry> {
    let memory_dir = root.join(".cold").join("memory");
    if !memory_dir.is_dir() {
        return Vec::new();
    }

    let Ok(entries) = std::fs::read_dir(&memory_dir) else {
        return Vec::new();
    };

    let mut result: Vec<MemoryEntry> = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension() else {
            continue;
        };
        if ext != "md" {
            continue;
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let content = read_memory_file(&path, &memory_dir);
        if !content.is_empty() {
            result.push(MemoryEntry { name, content });
        }
    }

    // Sort by name for deterministic ordering.
    result.sort_by(|a, b| a.name.cmp(&b.name));
    result
}

/// Build the memory section for injection into the system prompt.
#[must_use]
pub fn build_memory_prompt(entries: &[MemoryEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }

    let mut parts = vec!["# Memory\nThe following are persistent memory entries:".to_string()];

    for entry in entries {
        parts.push(format!("## {}\n{}", entry.name, entry.content));
    }

    parts.join("\n\n")
}

/// Read a memory file, resolving `@include:filename.md` directives.
fn read_memory_file(path: &Path, memory_dir: &Path) -> String {
    let Ok(meta) = std::fs::metadata(path) else {
        return String::new();
    };

    #[allow(clippy::cast_possible_truncation)]
    let max_bytes = MAX_MEMORY_FILE_BYTES as usize;

    let raw = if meta.len() > MAX_MEMORY_FILE_BYTES {
        let Ok(bytes) = std::fs::read(path) else {
            return String::new();
        };
        String::from_utf8_lossy(&bytes[..max_bytes.min(bytes.len())]).into_owned()
    } else {
        let Ok(content) = std::fs::read_to_string(path) else {
            return String::new();
        };
        content
    };

    // Resolve @include directives (one level deep only).
    resolve_includes(&raw, memory_dir)
}

/// Replace `@include:filename.md` with the contents of the referenced file.
///
/// Only resolves one level of includes to prevent infinite recursion.
fn resolve_includes(content: &str, memory_dir: &Path) -> String {
    let mut result = String::with_capacity(content.len());

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(filename) = trimmed.strip_prefix("@include:") {
            let filename = filename.trim();
            let include_path = memory_dir.join(filename);
            if include_path.is_file() {
                if let Ok(included) = std::fs::read_to_string(&include_path) {
                    result.push_str(&included);
                    result.push('\n');
                    continue;
                }
            }
            // If include fails, keep the directive line as-is.
        }
        result.push_str(line);
        result.push('\n');
    }

    // Remove trailing newline added by the loop.
    if result.ends_with('\n') {
        result.pop();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_memory_prompt_empty() {
        assert!(build_memory_prompt(&[]).is_empty());
    }

    #[test]
    fn test_build_memory_prompt() {
        let entries = vec![
            MemoryEntry {
                name: "preferences".into(),
                content: "I like Rust.".into(),
            },
            MemoryEntry {
                name: "context".into(),
                content: "Working on cold-agent-sdk.".into(),
            },
        ];
        let prompt = build_memory_prompt(&entries);
        assert!(prompt.contains("# Memory"));
        assert!(prompt.contains("## preferences"));
        assert!(prompt.contains("I like Rust."));
        assert!(prompt.contains("## context"));
    }

    #[test]
    fn test_resolve_includes_no_directives() {
        let content = "Hello world\nNo includes here";
        let result = resolve_includes(content, Path::new("/tmp"));
        assert_eq!(result, content);
    }

    #[test]
    fn test_load_memory_files_missing_dir() {
        let entries = load_memory_files(Path::new("/nonexistent/path"));
        assert!(entries.is_empty());
    }
}
