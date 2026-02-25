import os
import re
import time
import json
from google import genai
from datetime import datetime, timedelta, date

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("⚠️  pytrends not installed — keyword trend scoring will be skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATOR
# Runs before any AI call or API call.
# Catches configuration problems early so the pipeline doesn't waste
# Gemini quota on a run that was going to fail anyway.
# Returns {"valid": bool, "errors": [], "warnings": []}
#   errors   → hard blocks: pipeline stops
#   warnings → soft issues: pipeline continues but logs them
# ══════════════════════════════════════════════════════════════════════════════

def validate_planner_inputs(sections_config, existing_titles, config):
    errors   = []
    warnings = []

    # ── Validate sections_config ──────────────────────────────────────────────
    if not sections_config or not isinstance(sections_config, list):
        errors.append("sections_config must be a non-empty list of section dicts.")
    else:
        for i, section in enumerate(sections_config):
            if not section.get("Name"):
                errors.append(f"sections_config[{i}] is missing 'Name'.")
            if not section.get("Description"):
                warnings.append(
                    f"sections_config[{i}] ('{section.get('Name', 'unknown')}') "
                    "has no Description — topic quality may be reduced."
                )

    # ── Validate existing_titles ──────────────────────────────────────────────
    if not isinstance(existing_titles, list):
        errors.append("existing_titles must be a list (can be empty []).")

    # ── Validate Config_Brand fields ──────────────────────────────────────────
    brand = config.get("brand", {})

    if not brand.get("brand_name"):
        warnings.append(
            "brand_name missing from Config_Brand — "
            "Gemini will use generic brand references."
        )

    # NOTE: target_audience was removed — replaced by more specific fields below.
    # These give Gemini much more precise instructions for content generation.
    if not brand.get("audience_age_range"):
        warnings.append(
            "audience_age_range missing — Gemini may generate off-demographic topics."
        )
    if not brand.get("audience_gender"):
        warnings.append(
            "audience_gender missing — audience targeting will be generic."
        )
    if not brand.get("audience_pain_points"):
        warnings.append(
            "audience_pain_points missing — topic relevance may be reduced."
        )
    if not brand.get("competitor_names"):
        warnings.append(
            "competitor_names missing — competitor topic avoidance will not be enforced."
        )
    if not brand.get("content_language"):
        warnings.append(
            "content_language missing from Config_Brand — defaulting to English (en)."
        )
    if not brand.get("industry"):
        warnings.append(
            "industry missing from Config_Brand — compliance rules may not load correctly."
        )
    if not config.get("planner", {}).get("trend_score_threshold"):
        warnings.append(
            "trend_score_threshold missing from Config_Planner — defaulting to 10."
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC VALIDATOR
# Runs on each AI-generated topic BEFORE adding it to the pipeline.
# This is a fast, rule-based check (no AI, no API calls).
# The goal: reject vague, duplicate, or low-quality ideas early.
# ══════════════════════════════════════════════════════════════════════════════

def validate_topic(idea, existing_titles_lower):
    """
    Hard-validates a single generated topic dict.
    Returns {"valid": bool, "reason": str}

    existing_titles_lower: list of existing titles already lowercased,
    for fast similarity comparison.
    """
    title   = idea.get("Title",   "").strip()
    keyword = idea.get("Keyword", "").strip()
    summary = idea.get("Summary", "").strip()

    # ── Basic field checks ────────────────────────────────────────────────────
    if not title:
        return {"valid": False, "reason": "Title is empty."}

    word_count = len(title.split())
    if word_count < 5 and "?" not in title:
        return {
            "valid":  False,
            "reason": f"Title too vague ({word_count} words, no question): '{title}'"
        }

    if not keyword or len(keyword.strip()) < 2:
        return {"valid": False, "reason": "Keyword is too short or empty."}

    if not summary or len(summary.strip()) < 10:
        return {"valid": False, "reason": "Summary is too short or empty."}

    # ── Duplicate similarity check ────────────────────────────────────────────
    # We strip punctuation before comparing so "why does x cause y?"
    # matches "why does x cause y" correctly.
    title_lower = title.lower()
    clean_title = re.sub(r'[^\w\s]', '', title_lower)
    title_words = set(clean_title.split())

    # Guard: if no valid words remain after cleaning, reject
    if not title_words:
        return {
            "valid":  False,
            "reason": "Title contains no valid words after cleaning."
        }

    for existing in existing_titles_lower:
        clean_existing = re.sub(r'[^\w\s]', '', existing)
        existing_words = set(clean_existing.split())

        # Skip empty/corrupt existing titles
        if not existing_words:
            continue

        # If more than 60% of the existing title's words appear in the new title,
        # consider it too similar and reject it.
        overlap = len(title_words & existing_words) / len(existing_words)
        if overlap > 0.6:
            return {
                "valid":  False,
                "reason": f"Title too similar to existing: '{existing}'"
            }

    return {"valid": True, "reason": ""}


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULING HELPER
# Computes the publish date for each post.
# Reads the preferred publish days from Config_Cadence (e.g. Mon/Wed/Fri).
# Also respects blackout_dates (e.g. holidays).
# ══════════════════════════════════════════════════════════════════════════════

# Maps day names (as typed in the Sheet) to Python weekday numbers
# Monday=0, Tuesday=1, ..., Sunday=6
DAY_NAME_TO_NUMBER = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}

def parse_publish_days(cadence_config):
    """
    Reads publish_days from Config_Cadence and converts to weekday numbers.
    Example: "Monday, Wednesday, Friday" → [0, 2, 4]
    Falls back to [0, 2, 4] (Mon/Wed/Fri) if not configured.
    """
    raw = cadence_config.get("publish_days", "Monday, Wednesday, Friday")
    day_numbers = []
    for day in raw.split(","):
        day_clean = day.strip().lower()
        if day_clean in DAY_NAME_TO_NUMBER:
            day_numbers.append(DAY_NAME_TO_NUMBER[day_clean])
        else:
            print(f"   ⚠️  Unknown publish day '{day.strip()}' — skipping.")

    # If nothing parsed correctly, fall back to Mon/Wed/Fri
    return sorted(day_numbers) if day_numbers else [0, 2, 4]


def compute_scheduled_date(base_date, post_index, cadence_config):
    """
    Calculates the target publish date for a post.

    base_date      : date object — usually today
    post_index     : 0-based position of this post in the current batch
                     (used to spread posts across different days)
    cadence_config : the full Config_Cadence dict from the Sheet

    Logic:
    - Gets publish days from Config_Cadence (e.g. [0, 2, 4] = Mon/Wed/Fri)
    - Cycles through them in order as posts are scheduled
    - Skips any dates listed in blackout_dates
    - Returns date as "YYYY-MM-DD" string
    """
    publish_days = parse_publish_days(cadence_config)

    # Parse blackout dates — e.g. "2026-12-25, 2026-01-01"
    raw_blackout  = cadence_config.get("blackout_dates", "")
    blackout_set  = set(
        d.strip() for d in raw_blackout.split(",") if d.strip()
    )

    # Determine which day slot and week this post falls into
    slot         = post_index % len(publish_days)
    week_offset  = post_index // len(publish_days)
    target_wday  = publish_days[slot]

    # Start from base_date + week offset, then find the correct weekday
    start   = base_date + timedelta(weeks=week_offset)
    days_fwd = (target_wday - start.weekday()) % 7
    target  = start + timedelta(days=days_fwd)

    # Skip blackout dates — advance by one day at a time until clear
    while target.strftime("%Y-%m-%d") in blackout_set:
        target += timedelta(days=1)

    return target.strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════════
# PLANNER AGENT
# Agent 1 of the pipeline.
#
# Responsibilities:
#   1. Ask Gemini to brainstorm blog topic ideas per section
#   2. Validate each idea (quality, duplicates)
#   3. Check Google Trends popularity for the keyword
#   4. Assign a scheduled publish date
#   5. Return the approved topics to the Orchestrator
#
# The Orchestrator then does a SECOND duplicate check against ChromaDB
# (semantic similarity) before saving to the Sheet.
# ══════════════════════════════════════════════════════════════════════════════

class PlannerAgent:

    def __init__(self, config=None):
        """
        Initializes the Planner Agent.
        Reads API keys and model settings from environment variables and config.
        """
        print("🧠 Initializing Planner Agent...")
        self.config = config or {"brand": {}, "planner": {}, "system": {}}

        # ── Gemini client ─────────────────────────────────────────────────────
        # Model priority: Config_System tab → GEMINI_MODEL env var → default
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("   ❌ GOOGLE_API_KEY missing — Planner will fail.")

        self.client = genai.Client(api_key=api_key)

        # Read the Gemini model name from Config_System if available,
        # otherwise fall back to the environment variable, then a hardcoded default
        self.model_name = (
            self.config.get("system", {}).get("Gemini_Model")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        print(f"   🤖 Model: {self.model_name}")

        # ── Pytrends client (Google Trends) ───────────────────────────────────
        # hl = language for trend results (read from locale, e.g. "en-US", "es-MX")
        locale = self.config.get("brand", {}).get("locale", "en-US")

        if PYTRENDS_AVAILABLE:
            try:
                self.pytrends = TrendReq(hl=locale, tz=360)
                print(f"   📈 Pytrends initialized (locale: {locale}).")
            except Exception as e:
                print(f"   ⚠️  Pytrends init failed: {e}")
                self.pytrends = None
        else:
            self.pytrends = None

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL: Keyword Trend Score
    # Calls Google Trends to check how popular a keyword is.
    # Returns a scored result — not just True/False — so we can log the score.
    #
    # Rate limiting note:
    #   Google Trends has strict rate limits. If we get a 429 (Too Many Requests),
    #   we immediately STOP all further trend checks for this run.
    #   Retrying a 429 only makes the ban longer — so we fail gracefully.
    #   The keyword is still accepted (passed=True) but flagged for manual review.
    # ──────────────────────────────────────────────────────────────────────────

    def check_keyword_popularity(self, keyword, retries=3):
        """
        Checks Google Trends interest score for a keyword over the past 12 months.

        Returns a dict:
          {
            "passed":  bool   — True if score ≥ threshold (or if check skipped)
            "score":   float  — avg interest score (0–100), or -1 if unavailable
            "flagged": bool   — True if result needs manual verification
            "reason":  str    — human-readable explanation
          }
        """
        # Read the minimum score threshold from Config_Planner (default: 10)
        threshold = int(
            self.config.get("planner", {}).get("trend_score_threshold", 10)
        )

        # If pytrends is not available or was disabled by a 429, skip gracefully
        if not self.pytrends:
            return {
                "passed":  True,
                "score":   -1,
                "flagged": True,
                "reason":  "Pytrends unavailable — trend score skipped."
            }

        # Read target country from Config_Brand (e.g. "US", "MX")
        target_country = self.config.get("brand", {}).get("target_country", "US")
        print(f"   📈 Checking Google Trends: '{keyword}' ({target_country})...")

        for attempt in range(retries):
            try:
                self.pytrends.build_payload(
                    [keyword],
                    cat       = 0,
                    timeframe = "today 12-m",
                    geo       = target_country,
                    gprop     = ""
                )
                data = self.pytrends.interest_over_time()

                # Always wait after a successful call to avoid triggering rate limits
                time.sleep(5 + (attempt * 2))

                if data.empty or keyword not in data.columns:
                    return {
                        "passed":  False,
                        "score":   0,
                        "flagged": False,
                        "reason":  f"No trend data found for '{keyword}'."
                    }

                avg_score = round(float(data[keyword].mean()), 2)
                passed    = avg_score >= threshold
                print(
                    f"      Score: {avg_score:.1f} "
                    f"(threshold: {threshold}) → "
                    f"{'✅ PASS' if passed else '❌ LOW'}"
                )
                return {
                    "passed":  passed,
                    "score":   avg_score,
                    "flagged": False,
                    "reason":  "" if passed else
                               f"Score {avg_score} is below threshold {threshold}."
                }

            except Exception as e:
                error_msg = str(e)

                # 429 = Google Trends rate ban.
                # Setting self.pytrends = None disables ALL further trend checks
                # for this run. Next call to check_keyword_popularity will hit the
                # "if not self.pytrends" guard at the top and return passed=True.
                if "429" in error_msg or "TooManyRequests" in error_msg.replace(" ", ""):
                    print(
                        "   🚨 Pytrends 429 rate limit hit — "
                        "disabling trend checks for this run."
                    )
                    self.pytrends = None
                    return {
                        "passed":  True,
                        "score":   -1,
                        "flagged": True,
                        "reason":  "Pytrends 429 rate limit — requires manual verification."
                    }

                # Other errors → exponential backoff: 10s, 20s, 40s
                wait = 2 ** attempt * 10
                print(
                    f"   ⚠️  Pytrends attempt {attempt + 1}/{retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        # All retries exhausted — fail open so the topic isn't discarded,
        # but flag it so you can manually check the keyword's popularity
        print(f"   ⚠️  Pytrends failed after {retries} retries. Flagging for review.")
        return {
            "passed":  True,
            "score":   -1,
            "flagged": True,
            "reason":  f"Pytrends failed after {retries} retries — manual verification needed."
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # Builds the Gemini prompt dynamically from Config_Brand fields.
    # This is what makes the system scalable across industries:
    # every variable (language, audience, compliance, tone) comes from the Sheet.
    # ──────────────────────────────────────────────────────────────────────────

    def _build_brainstorm_prompt(self, section_name, section_desc,
                                  existing_titles, amount, config):
        """
        Assembles the full Gemini prompt for topic brainstorming.
        All brand/audience/compliance context comes from Config_Brand.

        Parameters:
          section_name    : e.g. "Sleep"
          section_desc    : e.g. "Tips for better rest and recovery"
          existing_titles : list of already-published titles (to avoid duplicates)
          amount          : how many topics to request (we ask for extras as buffer)
          config          : full config dict from the Orchestrator
        """
        brand = config.get("brand", {})

        # ── Brand & audience context ──────────────────────────────────────────
        brand_name   = brand.get("brand_name",             "the brand")
        age_range    = brand.get("audience_age_range",     "38–55")
        gender       = brand.get("audience_gender",        "female")
        pain_points  = brand.get("audience_pain_points",   "fatigue, brain fog, weight gain")
        sophistication = brand.get("audience_sophistication", "educated, reads labels")
        tone         = brand.get("tone_formality",         "conversational")
        competitors  = brand.get("competitor_names",       "")
        products     = brand.get("product_names",          "")
        avoid_topics = brand.get("avoid_topics",           "competitor names, political topics")

        # ── Language & compliance ─────────────────────────────────────────────
        # content_language drives what language the output is written in
        language   = brand.get("content_language",    "en")
        industry   = brand.get("industry",            "supplements")
        compliance = brand.get("compliance_framework","FDA")
        disclaimer = brand.get("disclaimer_text",     "")

        # Map language code to a human-readable name for the prompt
        language_names = {
            "en": "English", "es": "Spanish", "pt": "Portuguese",
            "fr": "French",  "de": "German",  "it": "Italian"
        }
        language_label = language_names.get(language.lower(), language)

        # ── Existing titles block ─────────────────────────────────────────────
        titles_block = (
            "\n".join([f"- {t}" for t in existing_titles])
            if existing_titles else "None — this section is new."
        )

        # ── Compliance instructions ───────────────────────────────────────────
        # These change based on industry (FDA for supplements, FTC for fitness, etc.)
        compliance_rules = f"""
- Industry: {industry}
- Regulatory framework: {compliance}
- Never make disease claims (cure / treat / fix / reverse / prevent).
- Always use hedging language: "may support", "research suggests", "can help".
- Append disclaimer where appropriate: "{disclaimer}"
"""

        return f"""You are a senior SEO and content strategist for {brand_name}.

════════════════════════════════════════
LANGUAGE
════════════════════════════════════════
Write ALL output exclusively in {language_label}.
Titles, keywords, and summaries must all be in {language_label}.

════════════════════════════════════════
BRAND CONTEXT
════════════════════════════════════════
Products           : {products}
Section            : {section_name} — {section_desc}

════════════════════════════════════════
TARGET AUDIENCE
════════════════════════════════════════
Age range          : {age_range}
Gender             : {gender}
Key pain points    : {pain_points}
Sophistication     : {sophistication}
Tone               : {tone}

════════════════════════════════════════
COMPLIANCE RULES ({compliance})
════════════════════════════════════════
{compliance_rules}

════════════════════════════════════════
TASK
════════════════════════════════════════
Generate exactly {amount} original blog post idea(s).
Each must target a real search query this audience types into Google.

TITLE RULES:
- Phrase as a question this specific audience would Google.
  GOOD: "Why Does Perimenopause Cause Night Sweats — And What Actually Helps?"
  BAD:  "Managing Hot Flashes During Menopause"
- Minimum 8 words.
- Must reference a specific symptom, mechanism, or outcome — not a generic topic.
- Must NOT resemble any title in the existing list below.

KEYWORD RULES:
- Primary: one long-tail keyword (3–5 words) with clear informational search intent.
- Secondary: two related supporting keywords.

SUMMARY RULES:
- One sentence: what the article explains and who it helps.
- Must mention a specific biological mechanism, hormone, or symptom.

STRICT EXCLUSIONS:
- No disease claims.
- No competitor references: {competitors}.
- Avoid these topics entirely: {avoid_topics}.
- No generic wellness content with no {section_name.lower()} angle.

════════════════════════════════════════
EXISTING TITLES — Do NOT repeat or closely resemble:
════════════════════════════════════════
{titles_block}

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════
Output ONLY a valid JSON array inside a ```json block.
No text outside the block. No numbering. No pipe characters.

```json
[
  {{
    "Title": "Why Does Perimenopause Cause Night Sweats — And What Actually Helps?",
    "Keyword": "perimenopause night sweats causes",
    "SecondaryKeywords": "estrogen sleep disruption, menopause temperature regulation",
    "Summary": "Explains how fluctuating estrogen levels dysregulate the hypothalamus and trigger night sweats in perimenopausal women."
  }}
]
```"""

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # Called by the Orchestrator once per pipeline run.
    # ──────────────────────────────────────────────────────────────────────────

    def plan_next_posts(self, sections_config, existing_titles,
                        config=None, posts_per_section=1):
        """
        Orchestrates the full topic planning process for one pipeline run.

        Steps:
          1. Validate inputs
          2. For each section: ask Gemini for ideas
          3. Parse and validate each idea
          4. Check keyword popularity on Google Trends
          5. Assign scheduled publish dates
          6. Return approved topics to the Orchestrator

        Returns:
          {
            "status":   "SUCCESS" | "BLOCKED",
            "new_rows": [list of topic dicts ready for the Sheet],
            "warnings": [list of non-fatal messages],
            "errors":   [list of hard failure messages]
          }
        """
        cfg      = config or self.config
        errors   = []
        warnings = []
        new_rows = []

        # ── 1. VALIDATE INPUTS ────────────────────────────────────────────────
        validation = validate_planner_inputs(sections_config, existing_titles, cfg)
        if not validation["valid"]:
            return {
                "status":   "BLOCKED",
                "new_rows": [],
                "warnings": validation["warnings"],
                "errors":   validation["errors"]
            }
        warnings.extend(validation["warnings"])

        existing_lower = [t.lower() for t in existing_titles]
        today          = date.today()
        cadence_config = cfg.get("cadence", {})

        # ── 2. LOOP SECTIONS ──────────────────────────────────────────────────
        for section in sections_config:
            section_name = section.get("Name",        "")
            section_desc = section.get("Description", "")

            # Ask for more ideas than needed — we'll validate and pick the best.
            # Example: if posts_per_section=1, we request 5 candidates.
            # This gives us backups in case some fail validation.
            request_amount = posts_per_section * 5

            prompt = self._build_brainstorm_prompt(
                section_name    = section_name,
                section_desc    = section_desc,
                existing_titles = existing_titles,
                amount          = request_amount,
                config          = cfg
            )

            # ── 3. CALL GEMINI ────────────────────────────────────────────────
            try:
                response = self.client.models.generate_content(
                    model    = self.model_name,
                    contents = prompt
                )
                raw_text = response.text.strip()
            except Exception as e:
                errors.append(f"Gemini error for section '{section_name}': {e}")
                continue  # Skip this section, try next one

            # ── 4. PARSE JSON FROM GEMINI RESPONSE ───────────────────────────
            # The prompt asks Gemini to wrap output in a ```json ... ``` block.
            # We extract only that block to avoid parsing stray text.
            try:
                json_match = re.search(r"```json\s*(\[.*?\])\s*```",
                                       raw_text, re.DOTALL)
                if json_match:
                    ideas = json.loads(json_match.group(1))
                else:
                    # Fallback: maybe Gemini returned raw JSON without the code block
                    ideas = json.loads(raw_text)
            except (json.JSONDecodeError, AttributeError) as e:
                errors.append(
                    f"JSON parse failed for section '{section_name}': {e}\n"
                    f"Raw output (first 300 chars): {raw_text[:300]}"
                )
                continue

            # ── 5. VALIDATE + SELECT IDEAS ────────────────────────────────────
            chosen_count = 0

            for idea in ideas:
                if chosen_count >= posts_per_section:
                    break  # We have enough approved topics for this section

                # Fast rule-based check: rejects empty, vague, or duplicate titles
                check = validate_topic(idea, existing_lower)
                if not check["valid"]:
                    warnings.append(
                        f"Skipped idea '{idea.get('Title', '')}': {check['reason']}"
                    )
                    continue

                title   = idea.get("Title",   "").strip()
                keyword = idea.get("Keyword", "").strip()

                # ── 6. GOOGLE TRENDS CHECK ────────────────────────────────────
                # check_keyword_popularity returns:
                # {"passed": bool, "score": float, "flagged": bool, "reason": str}
                #
                # "flagged" = True means the check couldn't run (rate limit, error)
                # We still ACCEPT flagged keywords — just log them for manual review
                trend = self.check_keyword_popularity(keyword)

                if not trend["passed"] and not trend["flagged"]:
                    # Low score but check ran successfully → log warning, don't skip
                    # A low-trend topic can still be valuable for SEO (low competition)
                    warnings.append(
                        f"Low trend score for '{keyword}': {trend['reason']}"
                    )

                # Score of -1 means the check was skipped (pytrends unavailable)
                trend_score = trend["score"] if trend["score"] >= 0 else "N/A"
                # "TrendFlagged" in notes = needs manual keyword verification
                trend_note  = "TrendFlagged" if trend["flagged"] else ""

                # ── 7. ASSIGN PUBLISH DATE ────────────────────────────────────
                # post_index is based on total existing posts + already approved
                # in this run — ensures dates don't stack on the same day
                scheduled_date = compute_scheduled_date(
                    base_date      = today,
                    post_index     = len(existing_titles) + len(new_rows),
                    cadence_config = cadence_config
                )

                # Read scheduled time from Config_Cadence, default "09:00 AM ET"
                scheduled_time = cadence_config.get("publish_time", "09:00 AM ET")

                # ── 8. BUILD ROW DICT ─────────────────────────────────────────
                # This dict maps directly to the columns in the Content_Plan Sheet.
                # The Orchestrator's _planner_row_to_sheet_list() converts it
                # to the correct column order.
                new_rows.append({
                    "Status":            "Pending Approval",
                    "Title":             title,
                    "Section":           section_name,
                    "Keyword":           keyword,
                    "SecondaryKeywords": idea.get("SecondaryKeywords", ""),
                    "Summary":           idea.get("Summary",           ""),
                    "ScheduledDate":     scheduled_date,
                    "ScheduledTime":     scheduled_time,
                    # ReviewerNotes at this stage = only the trend flag (if any)
                    # Other agents will add their own notes later
                    "ReviewerNotes":     trend_note,
                })

                # Add the accepted title to our local duplicate list
                # so we don't generate a similar topic for the next section
                existing_lower.append(title.lower())
                chosen_count += 1

            # Warn if no valid topics were found for this section
            if chosen_count == 0:
                warnings.append(
                    f"Section '{section_name}': all {len(ideas)} generated ideas "
                    "failed validation — consider broadening the section description."
                )

        # ── FINAL RESULT ──────────────────────────────────────────────────────
        # BLOCKED = hard errors occurred AND no topics were generated at all
        # SUCCESS = at least some topics were generated (warnings are OK)
        status = "BLOCKED" if (errors and not new_rows) else "SUCCESS"

        return {
            "status":   status,
            "new_rows": new_rows,
            "warnings": warnings,
            "errors":   errors,
        }
