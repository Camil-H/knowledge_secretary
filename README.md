# knowledge_secretary

A $0, fully-automated daily digest. Once a day, via GitHub Actions, it:

1. **Newsletter** — assembles an industry newsletter from blogs, papers/preprints (PubMed, bioRxiv), and X accounts.
2. **YouTube** — summarizes new uploads from configured channels within a daily time window.
3. **Podcast** — generates a long, technical two-host podcast on a rotating topic, published to the static site with the audio embedded as a player.

Runs entirely on free tiers: free LLMs (LiteLLM with a live OpenRouter-free fallback), free TTS (Microsoft Edge), free data sources, and GitHub Actions on a public repo (unlimited minutes).

## How it works

- `src/core/` — shared spine: data contracts (`Item`/`Result`/`Context`), the plugin registries, dedup state (`state.py`), the `sources.yaml` loader (`sources_loader.py`), the OpenRouter LLM client (`llm.py`), and the `gather` driver. `src/fetchers/` holds the deterministic by-source-type clients (rss, url, youtube, x, pubmed, biorxiv); `src/delivery/` renders and publishes the page. Flow: **fetchers → task adapters → gather → task produce → delivery**.
- `src/tasks/<name>/` — one self-contained bucket per task (`__init__.py`, `prompt.md`, `adapters.py` for the framework's source/enricher code, `sources.yaml` for the committed source data). Adding a task is a new bucket; nothing else changes.
- Per-task source data (feeds, queries, handles, topics) lives in `src/tasks/<task>/sources.yaml`, committed directly to the repo (it's a public template and the source lists are non-sensitive). For newsletter and YouTube the file is a list of source-spec dicts (kinds: `feed`, `pubmed`, `biorxiv`, `twitter`, `yt_channel`; enrichers: `article_text`, `transcript`); for the podcast it's a top-level list of topic strings for its rotation. Adding a blog, channel, or topic is one line in that file.
- Dedup state lives in `state/seen.json`, committed back each run. Items are marked seen **only after successful delivery**, so a failed send never drops content.
- Every task's output is rendered to a single static page — last 7 days, newest first, older days collapsed behind `<details>` toggles — and published to GitHub Pages.
- Prompts are plain Markdown (`src/tasks/*/prompt.md`) — edit behavior without touching code.

## Run

```sh
uv sync
uv run python -m src.run [newsletter|youtube|podcast|all]
```

Scheduled by `.github/workflows/daily.yml` — podcast at 12:00 UTC (claims fresh LLM quota first), newsletter + YouTube at 13:05 UTC. `.github/workflows/ci.yml` runs ruff, ty, and pytest.

## Configuration

There's no central config file — framework knobs live as constants next to the code that uses them: model ranking in `src/core/llm.py`, and the page title / history depth / output dirs in `src/delivery/site.py`. Per-task sources live in each task's `sources.yaml` (below). The only runtime inputs are secrets, set as GitHub Actions repository secrets:

| Secret | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | all LLM calls (free `:free` models). A one-time $10 OpenRouter top-up raises the free cap to 1,000 req/day (20 RPM). |
| `PAGES_DEPLOY_TOKEN` | PAT with write access to the Pages repo (`Camil-H/camil-h.github.io`) so the workflow can publish the site cross-repo |
| `TWITTER_AUTH_TOKEN`, `TWITTER_CT0` | X/Twitter session tokens for the `twitter-cli` X source (optional; degrades to nothing if absent) |

The site publishes to `camilharoune.com/knowledge_secretary/` — a subpath of the personal site, deployed with `keep_files` so it never clobbers the homepage.

### Make it your own

This repo is a template — the committed source lists are one example (Camil's own blogs/channels/topics), kept only so the project runs out of the box. To use it as your own secretary, edit `src/tasks/<task>/sources.yaml` directly for each task: each ships with an example set you replace with your own. For newsletter and YouTube the file is a list of source-spec dicts; for the podcast it's a list of topic strings.

## Stack

Python 3.12 · uv · LiteLLM · feedparser · trafilatura · youtube-transcript-api · podcastfy + edge-tts · ruff · ty · pytest.

## Contributing

Personal project. Pull requests are welcome but are reviewed and require maintainer approval before merging.

## License

[MIT](LICENSE)
