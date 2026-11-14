//! Text templates in the syntax of Go's `text/template`, the subset that
//! `docker --format` is used with: `{{.Field}}`, `{{.A.B}}`, `{{.}}`, string
//! and integer literals, pipes such as `{{.Tag | upper}}`, `{{-` and `-}}`
//! to trim the space around an action, and Docker's functions: `json`,
//! `upper`, `lower`, `title`, `truncate`, `pad`, `join`, `split` and `len`.
//! As with Docker, `\t` and `\n` in the text between actions are a tab and a
//! newline. There is no `if` or `range`.

use anyhow::{Context, bail};
use serde_json::Value;

/// A parsed template, rendered against a JSON value as its `.`.
#[derive(Debug, Clone)]
pub struct Template {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    /// A pipeline: each command's result is the last argument of the next
    Action(Vec<Command>),
}

#[derive(Debug, Clone)]
enum Command {
    Operand(Operand),
    Call(Function, Vec<Operand>),
}

#[derive(Debug, Clone)]
enum Operand {
    /// `.` followed by field names, none for `.` itself
    Field(Vec<String>),
    Literal(Value),
}

#[derive(Debug, Clone, Copy)]
enum Function {
    Json,
    Upper,
    Lower,
    Title,
    Truncate,
    Pad,
    Join,
    Split,
    Len,
}

impl Function {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "json" => Self::Json,
            "upper" => Self::Upper,
            "lower" => Self::Lower,
            "title" => Self::Title,
            "truncate" => Self::Truncate,
            "pad" => Self::Pad,
            "join" => Self::Join,
            "split" => Self::Split,
            "len" => Self::Len,
            _ => return None,
        })
    }

    /// How many arguments it takes, a piped value included.
    fn arity(self) -> usize {
        match self {
            Self::Json | Self::Upper | Self::Lower | Self::Title | Self::Len => 1,
            Self::Truncate | Self::Join | Self::Split => 2,
            Self::Pad => 3,
        }
    }

    fn call(self, args: &[Value]) -> anyhow::Result<Value> {
        let count = |i: usize| -> anyhow::Result<usize> {
            match &args[i] {
                Value::Number(n) => n.as_u64().map(|n| n as usize),
                Value::String(s) => s.parse().ok(),
                _ => None,
            }
            .with_context(|| format!("expected a count, not {}", args[i]))
        };
        Ok(match self {
            Self::Json => serde_json::to_string(&args[0])?.into(),
            Self::Upper => text(&args[0]).to_uppercase().into(),
            Self::Lower => text(&args[0]).to_lowercase().into(),
            Self::Title => title(&text(&args[0])).into(),
            Self::Truncate => text(&args[0])
                .chars()
                .take(count(1)?)
                .collect::<String>()
                .into(),
            Self::Pad => format!(
                "{}{}{}",
                " ".repeat(count(1)?),
                text(&args[0]),
                " ".repeat(count(2)?)
            )
            .into(),
            Self::Join => match &args[0] {
                Value::Array(items) => {
                    let items: Vec<String> = items.iter().map(text).collect();
                    items.join(&text(&args[1])).into()
                }
                other => bail!("join takes a list, not {other}"),
            },
            Self::Split => {
                let sep = text(&args[1]);
                if sep.is_empty() {
                    bail!("split needs a separator");
                }
                text(&args[0])
                    .split(sep.as_str())
                    .collect::<Vec<_>>()
                    .into()
            }
            Self::Len => match &args[0] {
                Value::Array(items) => items.len(),
                Value::Object(fields) => fields.len(),
                Value::String(s) => s.chars().count(),
                other => bail!("len takes a list, object or string, not {other}"),
            }
            .into(),
        })
    }
}

/// A value as a template prints it: a string as itself, anything else as
/// JSON, and nothing as Go prints a missing value.
fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "<no value>".into(),
        other => other.to_string(),
    }
}

/// Each word with its first letter in upper case.
fn title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut start = true;
    for c in s.chars() {
        if start {
            out.extend(c.to_uppercase());
        } else {
            out.push(c);
        }
        start = !c.is_alphanumeric();
    }
    out
}

impl Template {
    pub fn parse(source: &str) -> anyhow::Result<Self> {
        let mut nodes = Vec::new();
        let mut rest = source;
        let mut trim_next = false;
        while !rest.is_empty() {
            let (text, action) = match rest.find("{{") {
                Some(i) => (&rest[..i], Some(&rest[i + 2..])),
                None => (rest, None),
            };
            let mut text = unescape(text);
            if trim_next {
                text = text.trim_start().to_string();
            }
            let Some(action) = action else {
                push_text(&mut nodes, text);
                break;
            };
            let end = action
                .find("}}")
                .with_context(|| format!("unclosed action at {:?}", preview(action)))?;
            let mut inner = &action[..end];
            if let Some(trimmed) = trim_marker(inner, true) {
                text = text.trim_end().to_string();
                inner = trimmed;
            }
            trim_next = false;
            if let Some(trimmed) = trim_marker(inner, false) {
                trim_next = true;
                inner = trimmed;
            }
            push_text(&mut nodes, text);
            nodes.push(Node::Action(
                parse_pipeline(inner).with_context(|| format!("in {{{{{inner}}}}}"))?,
            ));
            rest = &action[end + 2..];
        }
        Ok(Self { nodes })
    }

    /// The template with `data` as `.`.
    pub fn render(&self, data: &Value) -> anyhow::Result<String> {
        let mut out = String::new();
        for node in &self.nodes {
            match node {
                Node::Text(text) => out.push_str(text),
                Node::Action(pipeline) => {
                    let mut piped: Option<Value> = None;
                    for command in pipeline {
                        piped = Some(match command {
                            Command::Operand(operand) => evaluate(operand, data)?,
                            Command::Call(function, operands) => {
                                let mut args = operands
                                    .iter()
                                    .map(|o| evaluate(o, data))
                                    .collect::<anyhow::Result<Vec<_>>>()?;
                                args.extend(piped.take());
                                function.call(&args)?
                            }
                        });
                    }
                    out.push_str(&text(&piped.unwrap_or_default()));
                }
            }
        }
        Ok(out)
    }
}

fn push_text(nodes: &mut Vec<Node>, text: String) {
    if !text.is_empty() {
        nodes.push(Node::Text(text));
    }
}

/// `\t` and `\n` as a tab and a newline, as Docker reads its formats.
fn unescape(text: &str) -> String {
    text.replace("\\t", "\t").replace("\\n", "\n")
}

fn preview(text: &str) -> String {
    text.chars().take(20).collect()
}

/// The action without its trim marker, `- ` at its `start` or ` -` at its
/// end, if it has one: the space tells it from a negative number.
fn trim_marker(inner: &str, start: bool) -> Option<&str> {
    match start {
        true => inner
            .strip_prefix('-')
            .filter(|rest| rest.starts_with(char::is_whitespace)),
        false => inner
            .strip_suffix('-')
            .filter(|rest| rest.ends_with(char::is_whitespace)),
    }
}

#[derive(Debug, PartialEq)]
enum Token {
    Pipe,
    Word(String),
    Str(String),
}

fn tokenize(inner: &str) -> anyhow::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = inner.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '|' {
            chars.next();
            tokens.push(Token::Pipe);
        } else if c == '"' {
            chars.next();
            let mut s = String::new();
            loop {
                match chars.next().context("unclosed string")? {
                    '"' => break,
                    '\\' => s.push(match chars.next().context("unclosed string")? {
                        'n' => '\n',
                        't' => '\t',
                        other => other,
                    }),
                    other => s.push(other),
                }
            }
            tokens.push(Token::Str(s));
        } else {
            let mut word = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '|' || c == '"' {
                    break;
                }
                word.push(c);
                chars.next();
            }
            tokens.push(Token::Word(word));
        }
    }
    Ok(tokens)
}

fn parse_pipeline(inner: &str) -> anyhow::Result<Vec<Command>> {
    let tokens = tokenize(inner)?;
    if tokens.is_empty() {
        bail!("empty action");
    }
    let mut pipeline = Vec::new();
    for (i, command) in tokens.split(|t| *t == Token::Pipe).enumerate() {
        let Some((first, args)) = command.split_first() else {
            bail!("a pipe with no command on one side");
        };
        let piped = usize::from(i > 0);
        pipeline.push(match first {
            Token::Word(name) if Function::named(name).is_some() => {
                let function = Function::named(name).unwrap();
                let args = args.iter().map(operand).collect::<anyhow::Result<Vec<_>>>()?;
                if args.len() + piped != function.arity() {
                    bail!(
                        "{name} takes {} argument(s), not {}",
                        function.arity(),
                        args.len() + piped
                    );
                }
                Command::Call(function, args)
            }
            Token::Word(name) if !name.starts_with('.') && name.parse::<i64>().is_err() => {
                bail!("no function {name:?}: use json, upper, lower, title, truncate, pad, join, split or len")
            }
            _ if piped == 1 => bail!("only a function can take a piped value"),
            _ if !args.is_empty() => bail!("a value takes no arguments"),
            token => Command::Operand(operand(token)?),
        });
    }
    Ok(pipeline)
}

fn operand(token: &Token) -> anyhow::Result<Operand> {
    Ok(match token {
        Token::Str(s) => Operand::Literal(s.clone().into()),
        Token::Word(word) if word == "." => Operand::Field(Vec::new()),
        Token::Word(word) if word.starts_with('.') => {
            let fields: Vec<String> = word[1..].split('.').map(String::from).collect();
            if fields.iter().any(String::is_empty) {
                bail!("bad field {word:?}");
            }
            Operand::Field(fields)
        }
        Token::Word(word) => match word.parse::<i64>() {
            Ok(n) => Operand::Literal(n.into()),
            Err(_) => bail!("expected a value, not {word:?}"),
        },
        Token::Pipe => unreachable!("pipes split the commands"),
    })
}

fn evaluate(operand: &Operand, data: &Value) -> anyhow::Result<Value> {
    match operand {
        Operand::Literal(value) => Ok(value.clone()),
        Operand::Field(path) => {
            let mut value = data;
            for name in path {
                value = match value {
                    Value::Object(fields) => fields.get(name).with_context(|| {
                        let names: Vec<&str> = fields.keys().map(String::as_str).collect();
                        format!("no field .{name}: there are {}", names.join(", "))
                    })?,
                    other => bail!("no field .{name} in {other}"),
                };
            }
            Ok(value.clone())
        }
    }
}
