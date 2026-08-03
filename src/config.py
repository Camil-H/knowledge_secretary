# src/config.py
"""Every tunable knob for the whole app, in two parts.

"Yours to set" is what a fork changes on day one. Everything under "Defaults" already works;
those are the values to reach for when tuning behaviour, not when adopting the project.

Values only. What deliberately stays next to the code that uses it: compiled patterns, the
prompt files each task reads, the registries that map a kind to behaviour, and the
wire-protocol literals a provider dictates — none of those are things an operator tunes.

Readers import the module (`from src import config`) and reference `config.NAME`, so one
definition is what every caller and every test sees.
"""

import os

from src.core.models import ModelLimit, PartBudget

# == Yours to set =============================================================

# The rest of a fork's personalisation lives outside this file: sources in each task's
# sources.yaml, the editorial voice in the prompt markdown, schedule and publishing target in
# .github/workflows/daily.yml.

SITE_TITLE = "Knowledge Secretary"
SITE_SUBTITLE = "Daily newsletter, YouTube digest, and technical podcast"

# Voices must belong to TTS_LANGUAGE_CODE — Cloud TTS rejects a mismatch — and both hosts
# should come from the same voice family so the two sides of the conversation match.
TTS_LANGUAGE_CODE = "en-US"
TTS_VOICES: dict[str, str] = {
    "Person1": "en-US-Chirp3-HD-Iapetus",
    "Person2": "en-US-Chirp3-HD-Laomedeia",
}


# == Defaults =================================================================

# ----- HTTP -----

# One value for every caller, so it has to cover the slowest legitimate one: an LLM completion
# on a large free model runs 20-90s, while a feed fetch that hangs is bounded by its own task.
HTTP_TIMEOUT_S = 120

# Sent on the guarded fetch: publishers that served trafilatura's own request happily will 403
# a bare client, and the guard now issues the request itself.
HTTP_USER_AGENT = "Mozilla/5.0 (compatible; KnowledgeSecretary/1.0)"

ERROR_STATUS_FLOOR = 400
UNAUTHORIZED_STATUS = 401
FORBIDDEN_STATUS = 403
NOT_FOUND_STATUS = 404
RATE_LIMIT_STATUS = 429
SERVER_ERROR_STATUS = 500
SERVICE_UNAVAILABLE_STATUS = 503
GATEWAY_TIMEOUT_STATUS = 504

# a request that failed with one of these may well succeed on another attempt
TRANSIENT_STATUSES = frozenset(
    {RATE_LIMIT_STATUS, SERVER_ERROR_STATUS, SERVICE_UNAVAILABLE_STATUS, GATEWAY_TIMEOUT_STATUS}
)

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# Hops the guard walks itself, re-checking each one. An article sits one or two hops away
# (http -> https, then the canonical URL); a longer chain is a loop or a tracker.
MAX_REDIRECT_HOPS = 3

# ----- LLM -----

RATE_LIMIT_RETRIES = int(os.environ.get("LLM_RATE_LIMIT_RETRIES", "4"))
BACKOFF_START_S = 2
BACKOFF_CAP_S = 30

# ----- LLM: Google AI Studio -----

GEMINI_KEY_LABEL = "GOOGLE_AI_STUDIO_KEY"
GEMINI_AUTH_STATUSES = (UNAUTHORIZED_STATUS, FORBIDDEN_STATUS)
GEMINI_EST_OUTPUT_TOKENS = 2048

# quality-descending, so the 500-rpd model is the safety net rather than the default
GEMINI_TEXT_MODELS: list[ModelLimit] = [
    ModelLimit("gemini-3.6-flash", rpd=20, rpm=5, tpm=250_000),
    ModelLimit("gemini-3.5-flash", rpd=20, rpm=5, tpm=250_000),
    ModelLimit("gemini-3.5-flash-lite", rpd=20, rpm=5, tpm=250_000),
    ModelLimit("gemini-3.1-flash-lite", rpd=500, rpm=15, tpm=250_000),
]

# ----- LLM: OpenRouter -----

OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"
# wall-clock cap for the whole cascade so a many-item run can't burn minutes on backoff sleep
OPENROUTER_DEADLINE_S = 120.0

OPENROUTER_MODELS = [  # tried in order
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/openai/gpt-oss-20b:free",
]

# ----- Search -----

TAVILY_KEY_LABEL = "TAVILY_API_KEY"
TAVILY_SEARCH_DEPTH = "advanced"
TAVILY_MAX_RESULTS = 10
TAVILY_MIN_RESULTS = 5  # fewer usable pages than this is too thin to build an episode on
TAVILY_MAX_PAGE_CHARS = 4000
TAVILY_MAX_SOURCES_CHARS = 24000
TAVILY_RETRIES = 3
TAVILY_TIMEOUT_S = 30.0
# separate from the read budget because httpx retries connect per resolved address
TAVILY_CONNECT_TIMEOUT_S = 5.0

# ----- Fetching -----

LOOKBACK_HOURS = 48  # feed-scan window; dedup filters already-seen items on top
MAX_FETCH_WORKERS = 8

# ----- Newsletter -----

PUBMED_RETMAX = 30
OPENRXIV_MAX_PAGES = 20  # ~600 preprints; caps a busy window so one source can't stall the run
X_TWEET_LIMIT = 10  # ~2x the in-window mean per handle, so a busy handle still isn't truncated

NEWSLETTER_ITEM_CHAR_LIMIT = 20000
# Sized for the smallest model call() might fall back to (~32k tokens), not the selected one —
# we can't know which rung of the cascade answers.
NEWSLETTER_TOTAL_CHAR_BUDGET = 120000
NEWSLETTER_ITEM_CHAR_FLOOR = 1000

# ----- YouTube -----

YOUTUBE_TRANSCRIPT_CHAR_LIMIT = 12000
# Videos per summarization call. 5 × 12k chars ≈ 15k tokens, far under the 250k TPM ceiling, and
# it cuts both the request count and the rpm pacing sleep by five.
YOUTUBE_BATCH_SIZE = 5

# ----- Podcast: transcript -----

TRANSCRIPT_PARTS = 8  # 1 intro + 6 body + 1 outro
TRANSCRIPT_MAX_SOURCE_CHARS = 12000
TRANSCRIPT_MAX_TURN_CHARS = 1200
TRANSCRIPT_CONTEXT_TAIL_CHARS = 2500
TRANSCRIPT_INTRO_SOURCE_CHARS = 1200
TRANSCRIPT_RAW_LOG_CHARS = 400  # of an unusable part, so the next run says what came back

# Word targets sum to 5,550 — set by the Cloud TTS monthly character budget at 30 episodes a
# month, not by taste. Raising them costs money. max_tokens is only a ceiling: it bounds a
# runaway part without shortening one that respects its word target.
TRANSCRIPT_PART_BUDGETS: dict[str, PartBudget] = {
    "intro": PartBudget(words=250, max_tokens=1200),
    "body": PartBudget(words=800, max_tokens=2600),
    "outro": PartBudget(words=500, max_tokens=1800),
}

# ----- Podcast: audio -----

TTS_KEY_LABEL = "GOOGLE_CLOUD_TTS_KEY"
TTS_PCM_RATE_HZ = 24_000
TTS_RETRIES = 3
TTS_MAX_TURN_BYTES = 4500
TTS_MONTH_CHAR_BUDGET = 1_000_000
TTS_MP3_BITRATE = "32k"

# ----- Data retention -----

RETENTION_DAYS = 7

# ----- State -----

STATE_PATH = "state/seen.json"
LEDGER_PATH = "state/llm_ledger.json"

# ----- Delivery -----

HISTORY_DIR = "history"
OUT_DIR = "public"
