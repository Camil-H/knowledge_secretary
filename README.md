# knowledge_secretary

A $0, fully-automated daily digest. Once a day, via GitHub Actions, it:

1. **Newsletter** — assembles an industry newsletter from blogs, papers/preprints (PubMed, bioRxiv), and X accounts.
2. **YouTube** — summarizes new uploads from configured channels within a daily time window.
3. **Podcast** — generates a long, technical two-host podcast on the next topic from a queue (each topic used once), published to the static site with the audio embedded as a player.

Runs on free tiers: every LLM call — newsletter, YouTube, and the podcast transcript (podcastfy routes through LiteLLM) — uses OpenRouter's `:free` models; the podcast audio uses Google Cloud TTS (a monthly free quota covers a daily episode, or set `_TTS_MODEL = "edge"` in `src/tasks/podcast/task.py` for strictly $0 audio); plus free data sources and GitHub Actions on a public repo (unlimited minutes).

## How it works

The three products are independent daily tasks that share one shape: **gather → summarize → publish**.

- **Newsletter** — pulls new items from your blogs, journals and preprints (PubMed, bioRxiv), and X accounts, then an LLM writes them up, grouped into sections you define.
- **YouTube** — finds new uploads from your channels within the day's window and summarizes each from its transcript, falling back to the video description when no transcript is available.
- **Podcast** — takes the next topic from a queue, discovers sources for it, and generates a long two-host episode, published with an embedded audio player.

What each product *reads* is source data you control — one `sources.yaml` per task. How each product *writes* is driven by a plain-Markdown prompt per task. So adapting the digest to a different field is editing config and prose, not code: swap the sources, rewrite the prompts, rename the sections.

Every run renders to a single static page — the last 7 days, newest first, older days collapsed — published to GitHub Pages, and records what it has already seen so nothing repeats. Items are marked seen only after a successful publish, so a failed run never drops content.

## Run

```sh
uv sync
uv run python -m src.run [newsletter|youtube|podcast|all]
```

`.github/workflows/daily.yml` runs the tasks on a daily schedule; `.github/workflows/ci.yml` runs ruff, ty, and pytest.

## Configuration

There's no central config file — framework knobs live as constants next to the code that uses them: model ranking in `src/core/llm.py`, and the page title / history depth / output dirs in `src/delivery/site.py`. Per-task sources live in each task's `sources.yaml` (below). The only runtime inputs are secrets, set as GitHub Actions repository secrets:

| Secret | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | all LLM calls — newsletter, YouTube, podcast source-discovery, and the podcast transcript (via podcastfy/LiteLLM) — using free `:free` models. A one-time $10 OpenRouter top-up raises the free cap to 1,000 req/day (20 RPM). Required. |
| `GEMINI_API_KEY` | podcast text-to-speech (Google Cloud Text-to-Speech). Must be a **GCP API key with the Cloud Text-to-Speech API enabled**, not a Google AI Studio key. Required for the podcast unless you switch `_TTS_MODEL` to `edge`. |
| `PAGES_DEPLOY_TOKEN` | PAT with write access to the Pages repo (`Camil-H/camil-h.github.io`) so the workflow can publish the site cross-repo. Required. |
| `TWITTER_AUTH_TOKEN`, `TWITTER_CT0` | X/Twitter session tokens for the `twitter-cli` X source (optional; degrades to nothing if absent). |

The site publishes to `camilharoune.com/knowledge_secretary/` — a subpath of the owner's personal site, deployed with `keep_files` so it never clobbers the homepage.

### Make it your own

This repo is a template: the committed data is one example (the owner's blogs/channels/topics), kept only so it runs out of the box. Everything personal is a `sources.yaml` entry, a workflow value, or a branding constant. Fork it, then work down this checklist.

**1. Sources — what it reads.** Edit each `src/tasks/<task>/sources.yaml`:
- `newsletter/sources.yaml` — RSS feed URLs, PubMed queries, bioRxiv categories, X handles, and the `section:` names. Newsletter and YouTube files are lists of source-spec dicts (kinds: `feed`, `pubmed`, `biorxiv`, `twitter`, `yt_channel`).
- `youtube/sources.yaml` — channel IDs and their sections.
- `podcast/sources.yaml` — the topic queue (a list of strings, consumed one per run).
- If you rename sections, also update the section vocabulary in `src/tasks/newsletter/prompt.md`.

**2. Secrets.** Set the repository secrets in the table above.

**3. Publishing target.** In `.github/workflows/daily.yml`, both the `podcast` and `digest` jobs publish the site — repoint `external_repository`, `destination_dir`, and `PAGES_DEPLOY_TOKEN` to your own Pages repo. `keep_files: true` assumes you publish into a subpath of a larger site; drop it if the Pages repo is dedicated to this project. Podcast MP3s are hosted as GitHub Release assets of your own fork (needs `permissions: contents: write`, already set) — no change needed. The local build dir is `OUT_DIR` in `src/delivery/site.py`.

**4. Branding & editorial voice.** Site title/subtitle: `TITLE`/`SUBTITLE` in `src/delivery/site.py`. Per-task subject lines: `src/tasks/*/task.py`. Podcast name, tagline, and host roles: the `CONVERSATION_CONFIG` in `src/tasks/podcast/task.py`. The editorial framing (currently biotech/pharma) lives in the prompts — `src/tasks/newsletter/prompt.md`, `src/tasks/youtube/prompt.md`, `src/tasks/podcast/prompt.md`, and `src/tasks/podcast/source_discovery_prompt.md`. Update the `LICENSE` copyright line too.

**5. Schedule.** Cron times are in `.github/workflows/daily.yml` (UTC). The job `if:` guards key off the **exact** cron strings, so if you change a time you must update its matching `github.event.schedule == '...'` condition.

**6. Start clean.** Delete `state/seen.json` (dedup state + the owner's podcast-queue progress) and `history/*.json` (rendered digests from prior runs) so your first run starts fresh.

## Contributing

Personal project. Pull requests are welcome but are reviewed and require maintainer approval before merging.

## License

[MIT](LICENSE)
