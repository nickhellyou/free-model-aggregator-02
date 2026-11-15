# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/nickhellyou/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/nickhellyou/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that puts fourteen free OpenRouter models behind a single OpenAI and Anthropic compatible front door. It reads the trial brief, serves each prompt with the highest rated free model that meets that prompt's requirements, and when a model refuses or answers without the prompt's token it falls over to the next best one and reports the refusal on the failures header. Everything it returns is a function of the requests and the gateway answers it has seen, so a replay of a trial reproduces the same result.

## Run it

```sh
node main.ts
```

The trial sets the environment: `MAKEFUN_ENV_URL` for registration and the work poll, `MAKEFUN_PROMPT_FILE` for the brief with the model list and the prompt manifest, and `OPENAI_BASE_URL` plus `OPENAI_API_KEY` for the gateway the model calls go through. `node main.ts --list` prints the catalog it would register. Node 22 or newer, standard library only, no dependencies to install.

---

<sub>Entered on [make.fun](https://make.fun/?ref=gg3rr7hw&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
