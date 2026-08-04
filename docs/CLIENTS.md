# Clients and endpoints

locus speaks two HTTP wire protocols, so any tool that talks either
can use it as a drop-in local model server. Snapshot 2026-08-04;
verify env-var names against the client's current docs.

## Endpoints

| Method + path              | Protocol  | Notes                        |
|----------------------------|-----------|------------------------------|
| POST /v1/completions       | OpenAI    | text completion, SSE stream  |
| POST /v1/chat/completions  | OpenAI    | chat, SSE stream, tools      |
| POST /v1/embeddings        | OpenAI    | string or array input        |
| GET  /v1/models (+ /{id})  | OpenAI    | model discovery              |
| POST /v1/messages          | Anthropic | Messages API, SSE stream     |
| GET  /health               | -         | liveness                     |
| GET  /metrics              | -         | Prometheus text              |

Both chat endpoints accept a `tools` list and emit structured tool
calls -- OpenAI `tool_calls` / finish_reason `tool_calls`, Anthropic
`tool_use` / stop_reason `tool_use` -- in both non-streaming and
streaming responses. `logprobs`, `response_format` json_object
(constrained decoding), and `usage` accounting are supported on the
OpenAI paths.

## Clients

### OpenAI ecosystem
Point the client's base URL at locus (`http://host:port/v1`):

- OpenAI Python / JS SDKs: set `base_url` / `OPENAI_BASE_URL`.
- LangChain, LlamaIndex, LiteLLM: use their OpenAI-compatible
  provider with the base URL.
- Aider / Continue (OpenAI mode), Open WebUI: set the API base to
  locus.

Any API key is accepted and ignored (see Auth below).

### Anthropic ecosystem
- Anthropic Python / JS SDKs: set the base URL to locus.
- Claude Code: `ANTHROPIC_BASE_URL=http://host:port` runs Claude
  Code against a locally-served model.

### GitHub Copilot CLI (fully offline)
Since the 2026-04-07 Copilot CLI release it can target any OpenAI
Chat Completions endpoint:

    export COPILOT_PROVIDER_BASE_URL=http://host:port
    export COPILOT_MODEL=<the model name locus reports>
    export COPILOT_OFFLINE=true   # talk only to locus; telemetry off

Copilot CLI requires the model to support tool calling AND streaming
or it errors; locus provides both. (Copilot CLI may still require a
one-time GitHub sign-in even for local models.)

## Requirements for agentic clients

Agent CLIs (Copilot CLI, Claude Code, Aider) drive a tool loop, so
they need:

- Streaming: yes.
- Tool calling: yes, and now on the streaming paths too. Because
  locus recognizes a tool call only from the whole completion, a
  streamed request with `tools` BUFFERS and delivers the call (or
  the text) in the terminal chunk -- the client never sees raw tool
  JSON as incremental content.

Success also depends on the local MODEL producing clean tool calls;
locus renders the tool list into a system instruction and parses the
completion (best-effort), rather than native structured
function-calling. A capable instruct model is required for reliable
agent use.

## Caveats

- Auth: none yet. locus does not check an API key, so anything
  network-exposed is unauthenticated -- run it on localhost or behind
  a trusted proxy. Optional external-helper auth is planned.
- Tool execution: locus only EXPRESSES tool calls; it never runs
  them. Execution is the client's job (locus is a model runner, not
  an MCP host).
- Input richness: incoming image / tool_result content blocks are
  parsed but ignored -- text in, text out.
- Chat formatting: the chat endpoints apply the configured chat
  template; point real chat models at a template via
  chat::ChatTemplate::from_gguf.

See docs/COMPARISON.md for how locus positions against other engines.
