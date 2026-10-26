# Clients and endpoints

locus speaks the OpenAI and the Anthropic HTTP protocols, so a tool
that talks either can use it as a local model server. Client
environment-variable names change; check them against the client's
own docs.

## Endpoints

| Method and path               | Protocol  | Notes                                       |
|-------------------------------|-----------|---------------------------------------------|
| POST /v1/completions          | OpenAI    | text completion, SSE stream                 |
| POST /v1/chat/completions     | OpenAI    | chat, SSE stream, tools, image input        |
| POST /v1/embeddings           | OpenAI    | string or array input; CPU and CUDA only    |
| GET  /v1/models (and /{id})   | OpenAI    | model discovery                             |
| POST /v1/audio/speech         | OpenAI    | Orpheus-TTS; needs `--audio-decoder`        |
| POST /v1/images/generations   | OpenAI    | Janus-Pro; needs `--image-decoder`          |
| POST /v1/messages             | Anthropic | Messages API, SSE stream, tools, images     |
| GET  /health                  |           | liveness                                    |
| GET  /metrics                 |           | Prometheus text                             |

Both chat endpoints accept a `tools` list and return structured tool
calls (OpenAI `tool_calls`, Anthropic `tool_use`), streamed or not.
The OpenAI paths also support `logprobs`, `response_format:
json_object` (constrained decoding) and `usage` accounting.

Images in chat requests (an OpenAI `image_url` part with a base64
`data:` URL, or an Anthropic base64 image block) need a Janus-Pro
model whose GGUF holds the vision tower. locus never fetches a URL.
Requests with images do not support streaming, tools or JSON mode
yet (each is a 400).

## Clients

### OpenAI ecosystem

Point the client's base URL at locus (`http://host:port/v1`):

- OpenAI Python and JS SDKs: set `base_url` or `OPENAI_BASE_URL`.
- LangChain, LlamaIndex, LiteLLM: use their OpenAI-compatible
  provider with that base URL.
- Aider, Continue (OpenAI mode), Open WebUI: set the API base to
  locus.

Any API key is accepted (see Auth below).

### Anthropic ecosystem

- Anthropic Python and JS SDKs: set the base URL to locus.
- Claude Code: `ANTHROPIC_BASE_URL=http://host:port` runs it against a
  locally served model.

### GitHub Copilot CLI (offline)

Copilot CLI can target any OpenAI Chat Completions endpoint:

    export COPILOT_PROVIDER_BASE_URL=http://host:port
    export COPILOT_MODEL=<the model name locus reports>
    export COPILOT_OFFLINE=true   # talk only to locus; telemetry off

It requires a model with tool calling and streaming, and locus
provides both. It may still ask for a one-time GitHub sign-in, even
for local models.

## Agentic clients

Agent CLIs (Copilot CLI, Claude Code, Aider) drive a tool loop, so
they need streaming and tool calling. locus recognizes a tool call
only from the whole completion, so a streamed request with `tools` is
buffered: the call (or the text) arrives in the final chunk, and the
client never sees raw tool JSON as incremental content.

locus renders the tool list into a system instruction and parses the
completion (best effort); it has no native structured function
calling. Reliable agent use therefore needs a capable instruct model.

## Caveats

- Auth: `locus-server` does not check API keys, so anything exposed on
  a network is unauthenticated; run it on localhost or behind a
  trusted proxy. The server has an external-helper auth mechanism
  (`locus-auth-helper` is a reference helper with an allow-list), but
  the `locus-server` binary does not expose it yet.
- Tool execution: locus only expresses tool calls; it never runs them.
  Running them is the client's job (locus is a model runner, not an
  MCP host).
- Content blocks: `tool_use` and `tool_result` blocks in requests are
  dropped; only text (and images, with a vision model) reaches the
  model.
- Chat formatting: the chat endpoints use the chat template the GGUF
  carries, matched to a known family (DESIGN.md 4.5); an unknown
  template falls back to plain concatenation.

See docs/COMPARISON.md for how locus compares to other engines.
