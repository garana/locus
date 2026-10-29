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
| POST /v1/images/generations   | OpenAI    | Janus-Pro or PixArt-alpha; see Image output |
| POST /v1/messages             | Anthropic | Messages API, SSE stream, tools, images     |
| POST /v1/responses            | OpenAI    | Responses API, stateless, SSE stream, tools |
| GET  /health                  |           | liveness                                    |
| GET  /metrics                 |           | Prometheus text                             |

The chat endpoints and `/v1/responses` accept a `tools` list and
return structured tool calls (OpenAI `tool_calls`, Anthropic
`tool_use`, Responses `function_call` items), streamed or not.
The OpenAI paths also support `logprobs`, `response_format:
json_object` (constrained decoding) and `usage` accounting.

Images in chat requests (an OpenAI `image_url` part with a base64
`data:` URL, or an Anthropic base64 image block) need a Janus-Pro
model whose GGUF holds the vision tower. locus never fetches a URL.
Requests with images do not support streaming, tools or JSON mode
yet (each is a 400).

### Image output

`/v1/images/generations` returns one PNG as `b64_json`, from either
model the server loaded:

| Model           | Server flag                    | Image   | Extra fields (defaults)                                                    |
|-----------------|--------------------------------|---------|----------------------------------------------------------------------------|
| Janus-Pro-1B    | `--image-decoder <vq16.gguf>`  | 384x384 | `cfg` (5), `temperature` (1; 0 = argmax), `seed` (random)                  |
| PixArt-alpha    | `--image-model <dir>`          | 512x512 | `steps` (20, up to 100), `guidance_scale` (4.5), `seed`, `negative_prompt` |

A request whose `model` starts with `pixart` goes to PixArt, and so
does every request when no Janus model is loaded. `n` must be 1,
`size` must be the model's, and `response_format` must be `b64_json`
(others are a 400). The PixArt directory holds the three GGUFs from
`scripts/convert-t5-encoder.py`, `convert-pixart.py` and
`convert-sd-vae.py`. One image runs at a time; a second request while
one runs gets a 503 with `Retry-After`.

With the OpenAI SDK, pass the extra fields in `extra_body`:

```python
img = client.images.generate(
    model="pixart", prompt="a red fox in fresh snow", size="512x512",
    response_format="b64_json", extra_body={"steps": 20, "seed": 1})
```

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
  locally served model. Its first request is about 18k tokens (system
  prompt and tool list), so start the server with `--ctx 32768` or
  more.

### OpenAI Codex

Codex speaks only the Responses API. Add a provider to
`~/.codex/config.toml`:

    model_provider = "locus"
    model = "<the model name locus reports>"

    [model_providers.locus]
    name = "locus"
    base_url = "http://host:port/v1"
    wire_api = "responses"

Its first request is about 22k tokens (developer prompt and a long
tool list), so start the server with `--ctx 40960` or more, and use a
model that calls tools well. Codex sets no output limit and treats a
reply cut at the server's cap as an error, so raise the cap too
(`--max-tokens 8192`). Codex logs that it cannot decode the
`/v1/models` reply (it expects its own catalog format) and then uses
default model metadata; that is harmless.

### GitHub Copilot CLI (offline)

Copilot CLI can target any OpenAI Chat Completions endpoint:

    export COPILOT_PROVIDER_BASE_URL=http://host:port
    export COPILOT_MODEL=<the model name locus reports>
    export COPILOT_OFFLINE=true   # talk only to locus; telemetry off

It requires a model with tool calling and streaming, and locus
provides both. It may still ask for a one-time GitHub sign-in, even
for local models.

## Agentic clients

Agent CLIs (Copilot CLI, Claude Code, Codex, Aider) drive a tool loop, so
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
- Content arrays: on `/v1/chat/completions`, content may be a string
  or an array of parts (the OpenAI multimodal shape); text parts are
  joined with newlines. A part other than text or an image is a 400.
- Responses API: stateless, so `previous_response_id` is a 400 (send
  the whole conversation, as Codex does). `instructions` and every
  system or `developer` message join one leading system message;
  `function_call` and `function_call_output` items go back to the
  model as its JSON call and a "Tool result" user turn (an image in a
  tool's output becomes "[image omitted]"); reasoning and other
  foreign items are skipped. Only function tools are offered; a tool
  inside a `namespace` entry is shown to the model as
  `<namespace>.<name>` and called back under its own name. Image and
  file inputs from the user are a 400. With no `max_output_tokens`,
  the server's output cap applies (`--max-tokens`, default 1024).
- Chat formatting: the chat endpoints use the chat template the GGUF
  carries, matched to a known family (DESIGN.md 4.5); an unknown
  template falls back to plain concatenation.

See docs/COMPARISON.md for how locus compares to other engines.
