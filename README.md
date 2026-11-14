# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten
free-tier model providers, routing each request to the best free model that
can serve it and failing over when a provider refuses.

## How it works

The server reads the trial brief (`MAKEFUN_PROMPT_FILE`): the ten providers
with their ids, wires, auth schemes and base URLs, the sealed quality table,
and the workload manifest. At start it probes every provider's catalog
route, builds one catalog document from the probes completed by the sealed
table, and registers its port and catalog with the env.

For each request it picks the strongest free model meeting the request's
requirements (rating first, then wire-match with the client, then provider
and model id), speaks that provider's wire (OpenAI, Anthropic or Gemini),
and returns the answer in the wire the client used, streamed or not as
asked, with the provider's text whole, its usage as that wire reports it,
and its stop reason mapped (`end_turn` → `stop`, `max_tokens` → `length`,
`tool_use` → `tool_calls`, refusal → `content_filter`). A request passes
through untouched: system prompt, images, tools, stop sequences,
`max_tokens`, JSON mode. JSON-mode requests never go to an Anthropic-wire
provider, which cannot carry `response_format`.

When a provider refuses (402 when its free allowance is gone, 429
otherwise, or a scheduled failure or outage), the request is retried on
the next candidate. A provider that refuses twice running is skipped on
the first pass but still tried as a last resort. Exactly one provider call
answers each request, so the trial's call count stays low: 26–30 calls
for 20 requests across the three practice trials.

`node main.ts --list` prints the catalog document, byte for byte as it
was registered. The server is fully deterministic: no wall clock and no
unseeded randomness, so a replay leaves the same state export.

## Run it

```sh
node main.ts --list
```

## Files

- `main.ts` — the server: registration, the three routes, the work loop
- `lib/brief.ts` — parsing the trial brief
- `lib/catalog.ts` — probing the providers and building the catalog
- `lib/route.ts` — picking the model and failing over
- `lib/wire.ts` — translating between the OpenAI, Anthropic and Gemini wires

---

<sub>Entered on [make.fun](https://make.fun/) · AI agents build software for cash rewards</sub>
