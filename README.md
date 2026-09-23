# faroazul-bot — FutureEval forecasting bot

Fully autonomous bot for the Metaculus FutureEval bot tournament (Fall 2026) and MiniBench.
Built on [forecasting-tools](https://github.com/Metaculus/forecasting-tools) and the
structure of the official [bot template](https://github.com/Metaculus/metac-bot-template).

## How it works
1. **Research** (in parallel): two web-search researchers from different providers via
   OpenRouter's `openrouter:web_search` tool, plus AskNews (latest news + archive) when its
   secrets exist. Fallbacks: web plugin, then a model-knowledge brief, then no research.
2. **Ensemble**: frontier models from Anthropic, OpenAI and Google, several runs each
   (`metabot/config.py`, profiles `premium` / `standard` / `economy`, chosen automatically
   from the remaining OpenRouter credit).
3. **Prompts**: resolution criteria read literally, explicit base rates, status quo,
   pre-mortem, open-question guard, 13 percentiles for numeric/date questions.
4. **Aggregation**: trimmed mean of log-odds (binary), trimmed linear pool (multiple
   choice), point-wise trimmed mean of CDFs (numeric, discrete, date); binary clipped to
   [2%, 98%].
5. **Publishing**: forecast + private comment through the Metaculus API. Public logs
   never show forecasts.

## Setup
Secrets (Settings → Secrets and variables → Actions): `METACULUS_TOKEN`,
`OPENROUTER_API_KEY`, and optionally `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET`
(or `ASKNEWS_API_KEY`).

```bash
pip install -r requirements.txt
python -m pytest -q                              # offline tests, no keys needed
python main.py --mode budget                     # credit and chosen profiles
python main.py --mode models                     # ping every model (a few cents)
python main.py --mode test --dry-run             # bot-testing-area, no publishing
python main.py --mode tournament                 # what the scheduled workflow runs
python main.py --mode coverage --days 7          # share of closed questions forecast
```

Workflows: `forecast.yaml` (every 10 min), `test_bot.yaml` (manual), `unit_tests.yaml`
(on push), `keepalive.yaml` (prevents GitHub from disabling schedules after 60 days).

Licences: see `THIRD_PARTY_NOTICES.md`.
