# knowledge_secretary

A $0, fully-automated daily digest. Once a day, via GitHub Actions, it:

1. **Newsletter** — assembles an industry newsletter from blogs, papers/preprints (PubMed, bioRxiv), and X accounts.
2. **YouTube** — summarizes new uploads from configured channels within a daily time window.
3. **Podcast** — generates a long, technical two-host podcast on a rotating topic and publishes it as an RSS feed.

Runs entirely on free tiers: free LLMs (LiteLLM with a live OpenRouter-free fallback), free TTS (Microsoft Edge), free data sources, and GitHub Actions on a public repo (unlimited minutes).

## How it works

- `src/core/` — shared spine: data contracts (`Item`/`Result`/`Context`), registries, dedup state, the tiered LLM client, source adapters + enrichers (`gather`), and deliverers.
- `src/tasks/<name>/` — one self-contained bucket per task (`__init__.py` + `prompt.md`). Adding a task is a new bucket; nothing else changes.
- Sources are declared **per task** in `config.yaml` (kinds: `feed`, `pubmed`, `biorxiv`, `twitter`; enrichers: `article_text`, `transcript`). Adding a blog or channel is one line.
- Dedup state lives in `state/seen.json`, committed back each run. Items are marked seen **only after successful delivery**, so a failed send never drops content.
- Prompts are plain Markdown (`src/tasks/*/prompt.md`) — edit behavior without touching code.

## Run

```sh
uv sync
uv run python -m src.run [newsletter|youtube|podcast|all]
```

Scheduled by `.github/workflows/daily.yml` — podcast at 12:00 UTC (claims fresh LLM quota first), newsletter + YouTube at 13:05 UTC. `.github/workflows/ci.yml` runs ruff, ty, and pytest.

## Configuration

Edit `config.yaml` (sources, per-task LLM tiers, delivery) and `topics.yaml` (podcast topic rotation). Secrets are referenced as `${ENV_VAR}` and set as GitHub Actions repository secrets:

| Secret | Purpose |
| --- | --- |
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` | free LLM tiers + fallback |
| `SMTP_HOST`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM` | newsletter / YouTube email |
| `PAGES_BASE_URL` | podcast RSS base URL (GitHub Pages) |
| `X_COOKIE` | X/Twitter session for the `twitter` backend (optional; degrades to nothing if absent) |

## Stack

Python 3.12 · uv · LiteLLM · feedparser · trafilatura · youtube-transcript-api · podcastfy + edge-tts · ruff · ty · pytest.

## Contributing

Personal project. Pull requests are welcome but are reviewed and require maintainer approval before merging.

## License

[MIT](LICENSE)
