//! Render chat messages through the Rust DeepSeek-V4 encoder, for the parity
//! check in `scripts/dsv4_parity.py`.
//!
//! Reads one JSON object per line on stdin -- `{"messages": [...],
//! "thinking_mode": "chat"}` -- and writes one JSON string per line: the
//! rendered prompt, or null when the encoder declines. Keeping the transport
//! line-delimited JSON means the comparison never has to guess where a
//! rendering ends, and rendered prompts are full of newlines.

use std::io::{self, BufRead, Write};

use infera_router::encoding_dsv4::encode_messages;
use serde_json::Value;

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: Value = serde_json::from_str(&line).expect("case parses");
        let messages = case["messages"].as_array().expect("messages array");
        let mode = case["thinking_mode"].as_str().unwrap_or("chat");
        let rendered = encode_messages(messages, mode);
        writeln!(out, "{}", serde_json::to_string(&rendered)?)?;
    }
    Ok(())
}
