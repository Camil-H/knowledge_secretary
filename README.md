# Knowledge Secretary

A $0, fully-automated daily digest, run by GitHub Actions and published to GitHub Pages:

1. **Newsletter** — new items from your blogs, papers/preprints (PubMed, bioRxiv) and X accounts, written up and grouped into sections you define.
2. **YouTube** — new uploads from your channels, summarized from their transcripts.
3. **Podcast** — a long two-host episode on the next topic from a queue, researched with a search-grounded model, published with an audio player.

Free tiers throughout: text prefers Google AI Studio's Gemini models and falls back to OpenRouter's `:free` ones, and audio uses Google Cloud TTS, whose 1M characters a month covers a daily episode of ~35 minutes.

## How it works

Three independent daily tasks sharing one shape: **gather → summarize → publish**. Each reads a `sources.yaml` and writes through a plain-Markdown prompt, so adapting the digest to another field is editing config and prose rather than code.

Each task records output to `history/` and remembers what it has seen, so nothing repeats. Publishing is a second phase that renders the last `RETENTION_DAYS` of history into one static page — splitting the two lets the podcast and newsletter jobs publish in either order without clobbering each other. Items are marked seen only after a successful publish, and a podcast topic is marked aired only once its episode exists, so a failed run retries rather than skipping.

Per-day LLM request quotas are metered locally in `state/llm_ledger.json`.

## Run

```sh
uv sync
uv run python -m src.run [newsletter|youtube|podcast|all]
uv run python -m src.delivery.site   # render history/ -> public/index.html
```

`.github/workflows/daily.yml` runs the tasks daily; `ci.yml` runs ruff, ty and pytest.

## Configuration

`src/config.py` holds every knob in two parts: **Yours to set** (site title and subtitle, TTS language and voices) and **Defaults**, which already work. Prompts, patterns and provider wire literals deliberately stay next to the code that uses them.

| Secret | Purpose |
| --- | --- |
| `GOOGLE_AI_STUDIO_KEY` | Preferred text tier, and the podcast's grounded research. **Required for the podcast** — no key, no source material, no episode. |
| `GOOGLE_CLOUD_TTS_KEY` | Podcast audio. A **GCP key with the Cloud Text-to-Speech API enabled**. **Required for the podcast.** |
| `OPENROUTER_API_KEY` | Fallback text tier on `:free` models. An independent credential, so a broken Google key degrades here instead of downing all three products. Required. |
| `PAGES_DEPLOY_TOKEN` | PAT with write access to the Pages repo, for cross-repo publishing. Required. |
| `TWITTER_AUTH_TOKEN`, `TWITTER_CT0` | X session tokens for the optional X source; degrades to nothing if absent. |


## Make it your own

The committed sources and topics are the owner's, kept only so the repo runs out of the box.

1. **Sources.** Edit each `src/tasks/<task>/sources.yaml`: feeds, journals, and X handles for the newsletter; channel IDs for YouTube; the topic queue for the podcast. Renaming a section means updating the vocabulary in `newsletter/prompt.md` too.
2. **Secrets and publishing.** Set the secrets above, plus the repository variables `PAGES_REPOSITORY` and `PAGES_DESTINATION_DIR` — both fall back to the owner's. Drop `keep_files: true` from `.github/actions/publish/action.yml` if your Pages repo is dedicated to this project.
3. **Voice.** Branding in config's "Yours to set"; editorial framing in the prompt markdown under `src/tasks/`. Update the `LICENSE` copyright line.
4. **Schedule.** Cron times live in `daily.yml`, and each job's `if:` guard matches the **exact** cron string — change a time and you must change its guard.
5. **Start clean.** Delete `state/*.json` and `history/*.json` so the first run starts fresh.

## Contributing

Personal project. Pull requests are welcome but require maintainer approval before merging.

## License

[MIT](LICENSE)
