# Free-Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/hafidzj134/makefun-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/hafidzj134/makefun-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A deterministic server that routes each client request to the best-rated **free**
model among the providers it discovers, translating between the OpenAI,
Anthropic, and Gemini wire formats in both directions. Built for
[make.fun](https://make.fun)'s
[“The best free AI model aggregator” bounty](https://make.fun/bounty/free-model-aggregator-01).

## How it works

- **Catalog**: on startup the server probes every provider named in its brief
  and serves the exact free-model catalog on all three API surfaces
  (`--list` prints the same catalog without starting the server).
- **Planning**: before answering, it ranks every free model by the brief's
  ratings and pre-plans the whole trial against each provider's counted-call
  budget, so top-rated models are spent where they earn the most.
- **Routing**: each request goes to the best planned model that satisfies its
  requirements (tools, vision, JSON mode, streaming). Requests are converted
  to the provider's wire format and responses — JSON or streamed SSE — are
  converted back to the client's, including stop-reason mapping and usage.
- **Failover**: a provider refusal (402/429) before any response byte is
  written moves the request to the next planned provider; a provider that
  keeps refusing is treated as spent.
- **Determinism**: no clocks, randomness, or environment-dependent ordering —
  the same trial replays byte-for-byte.

## Run

Requires Node 22 (the entry uses only the Node standard library and erasable
TypeScript syntax):

```sh
node main.ts          # register with the environment, serve, poll until exit
node main.ts --list   # print the provider/model catalog as JSON and exit
```

The server reads its brief (environment URL, token, provider list) from the
environment variables the trial harness provides.
