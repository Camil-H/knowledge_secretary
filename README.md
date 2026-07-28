# Knowledge Secretary

A $0, fully-automated daily digest. Once a day, via GitHub Actions, it:

1. **Newsletter** — assembles an industry newsletter from blogs, papers/preprints (PubMed, bioRxiv), and X accounts.
2. **YouTube** — summarizes new uploads from configured channels within a daily time window.
3. **Podcast** — generates a long, technical two-host podcast on the next topic from a queue (each topic used once), published to the static site with the audio embedded as a player.

Runs on free tiers: the newsletter, YouTube, and the podcast transcript use OpenRouter's `:free` models; the podcast's source material comes from Gemini 3.6 Flash with Google Search grounding, free for 5k prompts/month (one per episode); the podcast audio uses Google Cloud TTS (a monthly free quota covers a daily episode, or set `_TTS_MODEL = "edge"` in `src/tasks/podcast/task.py` for strictly $0 audio); plus free data sources and GitHub Actions on a public repo (unlimited minutes).

## How it works

The three products are independent daily tasks that share one shape: **gather → summarize → publish**.

- **Newsletter** — pulls new items from your blogs, journals and preprints (PubMed, bioRxiv), and X accounts, then an LLM writes them up, grouped into sections you define.
- **YouTube** — finds new uploads from your channels within the day's window and summarizes each from its transcript, falling back to the video description when no transcript is available.
- **Podcast** — takes the next topic from a queue, researches it with a search-grounded model, and generates a long two-host episode from that material, published with an embedded audio player. A topic is marked aired only once an episode exists, so a failed run retries it rather than skipping it.

What each product *reads* is source data you control — one `sources.yaml` per task. How each product *writes* is driven by a plain-Markdown prompt per task. So adapting the digest to a different field is editing config and prose, not code: swap the sources, rewrite the prompts, rename the sections.

Each task records its output to `history/` and records what it has already seen so nothing repeats. Publishing is a second phase: it renders the last 7 days of that history — newest first, older days collapsed — into a single static page on GitHub Pages. Splitting the two is what lets the podcast job and the newsletter+YouTube job publish in either order without overwriting each other's cards. Items are marked seen only after a successful publish, so a failed run never drops content.

## Run

```sh
uv sync
uv run python -m src.run [newsletter|youtube|podcast|all]
uv run python -m src.delivery.site   # render history/ -> public/index.html
```

`.github/workflows/daily.yml` runs the tasks on a daily schedule; `.github/workflows/ci.yml` runs ruff, ty, and pytest.

## Configuration

There's no central config file — framework knobs live as constants next to the code that uses them: model ranking in `src/core/llm.py`, the research model in `src/tasks/podcast/content_generator.py`, episode length (`_MAX_OUTPUT_TOKENS`, `_MAX_NUM_CHUNKS`) in `src/tasks/podcast/task.py`, and the history depth / output dirs in `src/delivery/site.py`. The page title/subtitle are also constants there, overridable via the `SITE_TITLE`/`SITE_SUBTITLE` env vars without touching code. Per-task sources live in each task's `sources.yaml` (below). The only runtime inputs are secrets, set as GitHub Actions repository secrets:

| Secret | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | newsletter, YouTube, and the podcast transcript, using free `:free` models. A one-time $10 OpenRouter top-up raises the free cap to 1,000 req/day (20 RPM). Required. |
| `GOOGLE_AI_STUDIO_KEY` | an **AI Studio** key, used for the podcast's grounded research (Gemini 3.6 Flash + Google Search) and preferred for the transcript (Gemini 3.1 Flash). **Required for the podcast** — without it there is no source material and no episode is produced. Must belong to the same project where the Generative Language API is enabled, and is a different key from `GEMINI_API_KEY`. Free-tier limits are per-key (5 RPM / 250k TPM / 20 requests per day); one episode spends roughly ten. |
| `GEMINI_API_KEY` | podcast text-to-speech (Google Cloud Text-to-Speech). Must be a **GCP API key with the Cloud Text-to-Speech API enabled**, not a Google AI Studio key. Required for the podcast unless you switch `_TTS_MODEL` to `edge`. |
| `PAGES_DEPLOY_TOKEN` | PAT with write access to the Pages repo (`Camil-H/camil-h.github.io`) so the workflow can publish the site cross-repo. Required. |
| `TWITTER_AUTH_TOKEN`, `TWITTER_CT0` | X/Twitter session tokens for the `twitter-cli` X source (optional; degrades to nothing if absent). |

Both Google keys are project-scoped, and the failure mode when they are crossed is confusing: a `403 … API_KEY_SERVICE_BLOCKED` or "Gemini API has not been used in project *N*" means the key belongs to a project where the Generative Language API is not enabled — enabling it elsewhere will not help. The error names the key's project *number*; compare it against your AI Studio project:

```sh
gcloud projects describe <ai-studio-project-id> --format='value(projectNumber)'
```

A mismatch means `GOOGLE_AI_STUDIO_KEY` holds the wrong key (typically the Cloud-TTS one).

The site publishes to `camilharoune.com/knowledge_secretary/` — a subpath of the owner's personal site, deployed with `keep_files` so it never clobbers the homepage.

### Make it your own

This repo is a template: the committed data is one example (the owner's blogs/channels/topics), kept only so it runs out of the box. Everything personal is a `sources.yaml` entry, a workflow value, or a branding constant. Fork it, then work down this checklist.

**1. Sources — what it reads.** Edit each `src/tasks/<task>/sources.yaml`:
- `newsletter/sources.yaml` — RSS feed URLs, PubMed queries, bioRxiv categories, X handles, and the `section:` names. Newsletter and YouTube files are lists of source-spec dicts (kinds: `feed`, `pubmed`, `biorxiv`, `twitter`, `yt_channel`).
- `youtube/sources.yaml` — channel IDs and their sections.
- `podcast/sources.yaml` — the topic queue (a list of strings, consumed one per run).
- If you rename sections, also update the section vocabulary in `src/tasks/newsletter/prompt.md`.

**2. Secrets.** Set the repository secrets in the table above.

**3. Publishing target.** Set the repository variables `PAGES_REPOSITORY` (`owner/repo`) and `PAGES_DESTINATION_DIR` to your own Pages repo/subpath — both jobs in `.github/workflows/daily.yml` pass them through to `.github/actions/publish`, which falls back to the owner's (`Camil-H/camil-h.github.io`, `knowledge_secretary`) if unset. Also set `PAGES_DEPLOY_TOKEN` (below). `keep_files: true` in `.github/actions/publish/action.yml` assumes you publish into a subpath of a larger site; drop it if the Pages repo is dedicated to this project. Podcast MP3s are hosted as GitHub Release assets of your own fork (needs `permissions: contents: write`, already set) — no change needed. The local build dir is `OUT_DIR` in `src/delivery/site.py`.

**4. Branding & editorial voice.** Site title/subtitle: the `SITE_TITLE`/`SITE_SUBTITLE` env vars (fall back to `TITLE`/`SUBTITLE` in `src/delivery/site.py` if unset). Per-task subject lines: `src/tasks/*/task.py`. Podcast host roles and voices: `src/tasks/podcast/transcript_prompt.md` and the `_TTS_CONFIG` in `src/tasks/podcast/task.py`. The editorial framing (currently biotech/pharma) lives in the prompts — `src/tasks/newsletter/prompt.md`, `src/tasks/youtube/prompt.md`, `src/tasks/podcast/transcript_prompt.md`, and `src/tasks/podcast/research_prompt.md`. Update the `LICENSE` copyright line too.

**5. Schedule.** Cron times are in `.github/workflows/daily.yml` (UTC). The job `if:` guards key off the **exact** cron strings, so if you change a time you must update its matching `github.event.schedule == '...'` condition.

**6. Start clean.** Delete `state/seen.json` (dedup state + the owner's podcast-queue progress) and `history/*.json` (rendered digests from prior runs) so your first run starts fresh.

## Contributing

Personal project. Pull requests are welcome but are reviewed and require maintainer approval before merging.

## License

[MIT](LICENSE)
