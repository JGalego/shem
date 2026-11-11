//! The Shemfile: Dockerfile syntax, read into stages of instructions.
//!
//! A stage whose `FROM` is `scratch` or a shem image is a shem stage. Any other
//! stage is an ordinary Docker stage, kept as text for `docker build`. See
//! `docs/shemfile.md`.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    pub from: String,
    pub name: Option<String>,
    pub instructions: Vec<Instruction>,
    /// The stage's logical lines, `FROM` included, for handing to Docker.
    pub lines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    Copy {
        from: Option<String>,
        sources: Vec<String>,
        dest: String,
    },
    Env(Vec<(String, String)>),
    Workdir(String),
    Entrypoint(Vec<String>),
    Cmd(Vec<String>),
    Label(Vec<(String, String)>),
    Arg {
        name: String,
        default: Option<String>,
    },
    Warm {
        until: Until,
        max_steps: Option<u64>,
    },
    /// `KERNEL path [command line]`: a VM image's kernel, in its filesystem.
    Kernel {
        path: String,
        cmdline: String,
    },
    /// Anything else, such as `RUN`: fine in a Docker stage, an error in a shem one.
    Other {
        keyword: String,
        rest: String,
    },
}

/// When `WARM` stops the guest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Until {
    Syscall(u32),
    Steps(u64),
    Ebreak,
    /// A VM's console output ends with these bytes.
    Output(Vec<u8>),
    /// A model session is primed, up to where the user's prompt goes.
    Prompt,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("Shemfile line {line}: {message}")]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

/// System calls `WARM --until=syscall:NAME` knows by name.
const SYSCALLS: &[(&str, u32)] = &[
    ("ioctl", 29),
    ("openat", 56),
    ("close", 57),
    ("pipe2", 59),
    ("getdents64", 61),
    ("read", 63),
    ("write", 64),
    ("readv", 65),
    ("writev", 66),
    ("exit", 93),
    ("exit_group", 94),
    ("waitid", 95),
    ("rt_sigsuspend", 133),
    ("brk", 214),
    ("clone", 220),
    ("execve", 221),
    ("mmap", 222),
    ("wait4", 260),
    ("getrandom", 278),
    ("statx", 291),
    ("clock_nanosleep", 407),
];

pub fn parse(text: &str) -> Result<Vec<Stage>, ParseError> {
    let mut stages: Vec<Stage> = Vec::new();
    for (line, logical) in logical_lines(text) {
        let fail = |message: String| ParseError { line, message };
        let (keyword, rest) = match logical.split_once(char::is_whitespace) {
            Some((keyword, rest)) => (keyword.to_ascii_uppercase(), rest.trim().to_string()),
            None => (logical.to_ascii_uppercase(), String::new()),
        };
        if keyword == "FROM" {
            let words: Vec<&str> = rest.split_whitespace().collect();
            let (from, name) = match words.as_slice() {
                [from] => (from.to_string(), None),
                [from, r#as, name] if r#as.eq_ignore_ascii_case("as") => {
                    (from.to_string(), Some(name.to_string()))
                }
                _ => return Err(fail("expected FROM image [AS name]".into())),
            };
            stages.push(Stage {
                from,
                name,
                instructions: Vec::new(),
                lines: vec![logical.clone()],
            });
            continue;
        }
        let stage = stages
            .last_mut()
            .ok_or_else(|| fail("expected FROM first".into()))?;
        stage.lines.push(logical.clone());
        let instruction = match keyword.as_str() {
            "COPY" => parse_copy(&rest).map_err(fail)?,
            "ENV" => Instruction::Env(parse_pairs(&rest, true).map_err(fail)?),
            "LABEL" => Instruction::Label(parse_pairs(&rest, false).map_err(fail)?),
            "WORKDIR" if !rest.is_empty() => Instruction::Workdir(rest),
            "ENTRYPOINT" => Instruction::Entrypoint(parse_command(&rest).map_err(fail)?),
            "CMD" => Instruction::Cmd(parse_command(&rest).map_err(fail)?),
            "ARG" => match rest.split_once('=') {
                Some((name, default)) => Instruction::Arg {
                    name: name.trim().into(),
                    default: Some(unquote(default.trim())),
                },
                None if !rest.is_empty() => Instruction::Arg {
                    name: rest,
                    default: None,
                },
                None => return Err(fail("expected ARG name[=default]".into())),
            },
            "WARM" => parse_warm(&rest).map_err(fail)?,
            "KERNEL" if !rest.is_empty() => {
                let (path, cmdline) = rest.split_once(char::is_whitespace).unwrap_or((&rest, ""));
                Instruction::Kernel {
                    path: path.to_string(),
                    cmdline: cmdline.trim().to_string(),
                }
            }
            _ => Instruction::Other { keyword, rest },
        };
        stage.instructions.push(instruction);
    }
    Ok(stages)
}

/// Lines with comments and blanks removed and `\` continuations joined, with
/// the number of the line each starts on.
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut lines = Vec::new();
    let mut current: Option<(usize, String)> = None;
    for (number, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if current.is_none() && (trimmed.is_empty() || trimmed.starts_with('#')) {
            continue;
        }
        if current.is_some() && trimmed.starts_with('#') {
            continue; // comment lines inside a continuation are dropped, as Docker does
        }
        let (start, mut text) = current.take().unwrap_or((number + 1, String::new()));
        match trimmed.strip_suffix('\\') {
            Some(part) => {
                text.push_str(part);
                text.push(' ');
                current = Some((start, text));
            }
            None => {
                text.push_str(trimmed);
                lines.push((start, text));
            }
        }
    }
    if let Some(last) = current {
        lines.push(last);
    }
    lines
}

/// `--flag=value` options at the start of `rest`, then the remaining words.
fn split_flags(rest: &str) -> (Vec<(&str, &str)>, &str) {
    let mut flags = Vec::new();
    let mut remaining = rest.trim_start();
    while let Some(after) = remaining.strip_prefix("--") {
        let end = after.find(char::is_whitespace).unwrap_or(after.len());
        let flag = &after[..end];
        let (name, value) = flag.split_once('=').unwrap_or((flag, ""));
        flags.push((name, value));
        remaining = after[end..].trim_start();
    }
    (flags, remaining)
}

fn parse_copy(rest: &str) -> Result<Instruction, String> {
    let (flags, rest) = split_flags(rest);
    let mut from = None;
    for (name, value) in flags {
        match name {
            "from" => from = Some(value.to_string()),
            _ => return Err(format!("COPY --{name} is not supported")),
        }
    }
    let mut words = if rest.starts_with('[') {
        serde_json::from_str::<Vec<String>>(rest).map_err(|e| format!("bad COPY array: {e}"))?
    } else {
        rest.split_whitespace().map(unquote).collect()
    };
    if words.len() < 2 {
        return Err("expected COPY source... destination".into());
    }
    let dest = words.pop().unwrap();
    Ok(Instruction::Copy {
        from,
        sources: words,
        dest,
    })
}

/// `KEY=VALUE` pairs, or with `legacy`, also `KEY VALUE`.
fn parse_pairs(rest: &str, legacy: bool) -> Result<Vec<(String, String)>, String> {
    let words = split_words(rest);
    if legacy && !words.is_empty() && !words[0].contains('=') {
        let (key, value) = rest
            .split_once(char::is_whitespace)
            .ok_or("expected KEY VALUE")?;
        return Ok(vec![(key.to_string(), unquote(value.trim()))]);
    }
    words
        .iter()
        .map(|word| {
            let (key, value) = word
                .split_once('=')
                .ok_or(format!("expected KEY=VALUE, got {word:?}"))?;
            Ok((key.to_string(), unquote(value)))
        })
        .collect()
}

/// Exec form (a JSON array) as is; shell form run by `/bin/sh -c`.
fn parse_command(rest: &str) -> Result<Vec<String>, String> {
    if rest.starts_with('[') {
        serde_json::from_str(rest).map_err(|e| format!("bad JSON array: {e}"))
    } else if rest.is_empty() {
        Err("expected a command".into())
    } else {
        Ok(vec!["/bin/sh".into(), "-c".into(), rest.into()])
    }
}

fn parse_warm(rest: &str) -> Result<Instruction, String> {
    let mut until = None;
    let mut max_steps = None;
    // Split with quotes in mind, for an output condition such as "~ # "
    for word in split_words(rest) {
        let flag =
            (word.strip_prefix("--")).ok_or(format!("unexpected {word:?} after WARM's options"))?;
        let (name, value) = flag.split_once('=').unwrap_or((flag, ""));
        match name {
            "until" => until = Some(parse_until(value)?),
            "max-steps" => {
                max_steps = Some(
                    value
                        .parse()
                        .map_err(|_| format!("bad --max-steps {value:?}"))?,
                )
            }
            _ => return Err(format!("WARM --{name} is not an option")),
        }
    }
    Ok(Instruction::Warm {
        until: until.ok_or("WARM needs --until")?,
        max_steps,
    })
}

fn parse_until(value: &str) -> Result<Until, String> {
    if value == "ebreak" {
        return Ok(Until::Ebreak);
    }
    if value == "prompt" {
        return Ok(Until::Prompt);
    }
    if let Some(steps) = value.strip_prefix("steps:") {
        return steps
            .parse()
            .map(Until::Steps)
            .map_err(|_| format!("bad step count {steps:?}"));
    }
    if let Some(text) = value.strip_prefix("output:") {
        let text = if text.starts_with('"') {
            serde_json::from_str::<String>(text).map_err(|e| format!("bad output {text}: {e}"))?
        } else {
            unquote(text)
        };
        if text.is_empty() {
            return Err("--until=output: needs the text to wait for".into());
        }
        return Ok(Until::Output(text.into_bytes()));
    }
    if let Some(name) = value.strip_prefix("syscall:") {
        if let Ok(number) = name.parse() {
            return Ok(Until::Syscall(number));
        }
        return SYSCALLS
            .iter()
            .find(|(known, _)| *known == name)
            .map(|&(_, number)| Until::Syscall(number))
            .ok_or(format!(
                "unknown system call {name:?}; give its number instead"
            ));
    }
    Err(format!(
        "expected --until=syscall:NAME, steps:N, ebreak, output:TEXT or prompt, got {value:?}"
    ))
}

/// Words split on whitespace, keeping quoted runs together.
fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    for c in text.chars() {
        match (quote, c) {
            (None, '"' | '\'') => {
                quote = Some(c);
                current.push(c);
            }
            (Some(q), c) if c == q => {
                quote = None;
                current.push(c);
            }
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            (_, c) => current.push(c),
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn unquote(text: &str) -> String {
    let text = text.trim();
    for q in ['"', '\''] {
        if text.len() >= 2 && text.starts_with(q) && text.ends_with(q) {
            return text[1..text.len() - 1].to_string();
        }
    }
    text.to_string()
}
