# DeepSeek-V4 chat-encoding golden pairs

Copied verbatim from `encoding/tests/` in the `deepseek-ai/DeepSeek-V4-Pro`
model repository (MIT, Copyright (c) 2023 DeepSeek), which is also the source of
`sglang.srt.entrypoints.openai.encoding_dsv4`. 40 KB total.

Each `test_input_N.json` / `test_output_N.txt` pair is a byte-exact oracle for
`crate::encoding_dsv4::encode_messages`, and they are consumed by the unit tests
in that module:

| pair | thinking_mode | covers |
| ---- | ------------- | ------ |
| 1 | `thinking` | system tools, DSML tool call, tool result merged into a user turn |
| 2 | `thinking` | `drop_thinking` removing an earlier turn's reasoning |
| 3 | `thinking` | `developer` role with tools, `latest_reminder`, CJK content |
| 4 | `chat`     | `latest_reminder`, the `action` task token |

Pair 1 keeps its tool list in a top-level `tools` key; the reference harness
assigns it to `messages[0]["tools"]` before encoding, and so does our test.
