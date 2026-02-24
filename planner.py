import os
import re
import time
import json
from google import genai
from datetime import datetime, timedelta

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("⚠️ pytrends not installed — keyword trend scoring will be skipped.")


# ─────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────

def validate_planner_inputs(sections_config, existing_titles, config):
    """
    Hard-gate validation before planning begins.
    Returns {"valid": bool, "errors": [], "warnings": []}
    """
    errors   = []
    warnings = []

    if not sections_config or not isinstance(sections_config, list):
        errors.append(
            "sections_config must be a non-empty list of section dicts."
        )
    else:
        for i, section in enumerate(sections_config):
            if not section.get("Name"):
                errors.append(f"sections_config[{i}] is missing 'Name'.")
            if not section.get("Description"):
                warnings.append(
                    f"sections_config[{i}] ('{section.get('Name', 'unknown')}') "
                    "has no Description — topic quality may be reduced."
                )

    if not isinstance(existing_titles, list):
        errors.append("existing_titles must be a list (can be empty []).")

    brand = config.get("brand", {})
    if not brand.get("brand_name"):
        warnings.append(
            "brand_name missing from Config_Brand — "
            "Gemini will use generic brand references."
        )
    if not brand.get("target_audience"):
        warnings.append(
            "target_audience missing — "
            "Gemini may generate off-demographic topics."
        )
    if not brand.get("competitor_names"):
        warnings.append(
            "competitor_names missing — "
            "competitor topic avoidance will not be enforced."
        )
    if not config.get("planner", {}).get("trend_score_threshold"):
        warnings.append(
            "trend_score_threshold missing from Config_Planner — "
            "defaulting to 10."
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ─────────────────────────────────────────────
# TOPIC VALIDATOR
# ─────────────────────────────────────────────

def validate_topic(idea, existing_titles_lower):
    """
    Hard-validates a single generated topic dict.
    Returns {"valid": bool, "reason": str}
    """
    title   = idea.get("Title",   "").strip()
    keyword = idea.get("Keyword", "").strip()
    summary = idea.get("Summary", "").strip()

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

    # ── Fix #2 + #3: clean punctuation before matching,
    #    guard against empty title_words to prevent ZeroDivisionError ──
    title_lower = title.lower()
    # Fix #3: r'[^\w\s]' — single backslash, correct regex in raw string
    clean_title = re.sub(r'[^\w\s]', '', title_lower)
    title_words = set(clean_title.split())

    # Fix #2: early return if title has no valid words after cleaning
    if not title_words:
        return {
            "valid":  False,
            "reason": "Title contains no valid words after cleaning."
        }

    for existing in existing_titles_lower:
        clean_existing = re.sub(r'[^\w\s]', '', existing)
        existing_words = set(clean_existing.split())

        # Guard: skip empty existing titles (bad data in sheet)
        if not existing_words:
            continue

        overlap = len(title_words & existing_words) / len(existing_words)
        if overlap > 0.6:
            return {
                "valid":  False,
                "reason": f"Title too similar to existing: '{existing}'"
            }

    return {"valid": True, "reason": ""}


# ─────────────────────────────────────────────
# SCHEDULING HELPER
# ─────────────────────────────────────────────

def compute_scheduled_date(base_date, post_index, posts_per_week=3):
    """
    Staggers post dates starting from base_date.
    Distributes posts evenly Mon / Wed / Fri.
    """
    publish_days = [0, 2, 4]
    week_offset  = post_index // len(publish_days)
    day_offset   = publish_days[post_index % len(publish_days)]
    target       = base_date + timedelta(weeks=week_offset, days=day_offset)
    return target.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────

class PlannerAgent:
    def __init__(self, config=None):
        """
        Agent 1: Content Planner & Strategist.
        """
        print("🧠 Initializing Planner Agent...")
        self.config = config or {"brand": {}, "planner": {}}

        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

        if PYTRENDS_AVAILABLE:
            try:
                self.pytrends = TrendReq(hl="en-US", tz=360)
                print("   📈 Pytrends initialized.")
            except Exception as e:
                print(f"   ⚠️ Pytrends init failed: {e}")
                self.pytrends = None
        else:
            self.pytrends = None

    # ── TOOL: Keyword Trend Score ────────────────────────────────────────
    def check_keyword_popularity(self, keyword, retries=3):
        """
        TOOL: Checks Google Trends interest for a keyword.
        Returns a scored result dict — not just True/False.

        Fix #1: 429 Too Many Requests exits immediately
        (retrying during a ban extends the ban).
        """
        threshold = int(
            self.config.get("planner", {}).get("trend_score_threshold", 10)
        )

        if not self.pytrends:
            return {
                "passed":  True,
                "score":   -1,
                "flagged": True,
                "reason":  "Pytrends unavailable — trend score skipped."
            }

        print(f"   📈 Checking Google Trends: '{keyword}'...")

        for attempt in range(retries):
            try:
                self.pytrends.build_payload(
                    [keyword], cat=0, timeframe="today 12-m", geo="US", gprop=""
                )
                data = self.pytrends.interest_over_time()

                # Sleep after every successful call to respect rate limits
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
                    f"{'✅ PASS' if passed else '❌ SKIP'}"
                )
                return {
                    "passed":  passed,
                    "score":   avg_score,
                    "flagged": False,
                    "reason":  "" if passed else
                               f"Score {avg_score} below threshold {threshold}."
                }

            except Exception as e:
                error_msg = str(e)

                # Fix #1: 429 means we are rate-banned — stop immediately
                # Retrying extends the ban. Fail open and flag for review.
                #
                # ── BONUS: what `self.pytrends = None` inside the 429 handler does ──
                # Setting `self.pytrends = None` on a 429 is intentional and important.
                # When the next topic calls `check_keyword_popularity`, the first guard triggers:
                # if not self.pytrends:
                #     return {"passed": True, "score": -1, "flagged": True, ...}
                if "429" in error_msg or "TooManyRequests" in error_msg.replace(" ", ""):
                    print(
                        f"   🚨 Pytrends 429 rate limit hit — "
                        "stopping all trend checks to avoid extending the ban."
                    )
                    # Disable pytrends for the rest of this run
                    self.pytrends = None
                    return {
                        "passed":  True,
                        "score":   -1,
                        "flagged": True,
                        "reason":  "Pytrends 429 rate limit — requires manual verification."
                    }

                # Other errors: exponential backoff and retry
                wait = 2 ** attempt * 10  # 10s → 20s → 40s
                print(
                    f"   ⚠️ Pytrends attempt {attempt + 1}/{retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        # All retries exhausted — fail open with flag
        print(f"   ⚠️ Pytrends failed after {retries} retries. Flagging for review.")
        return {
            "passed":  True,
            "score":   -1,
            "flagged": True,
            "reason":  f"Pytrends failed after {retries} retries — requires manual verification."
        }

    # ── PROMPT BUILDER ────────────────────────────────────────────────────
    def _build_brainstorm_prompt(self, section_name, section_desc,
                                  existing_titles, amount, config):
        brand       = config.get("brand", {})
        brand_name  = brand.get("brand_name",       "Glomend")
        audience    = brand.get("target_audience",  "women aged 38–55 navigating perimenopause and menopause")
        competitors = brand.get("competitor_names", "Goli, Bonafide, Ritual, Menofit, Equelle")
        products    = brand.get("product_names",    "GloRest, GloSerene, GloBalance")

        titles_block = "\n".join([f"- {t}" for t in existing_titles]) \
            if existing_titles else "None — this section is new."

        return f"""You are a senior SEO and content strategist for {brand_name},
a women's health supplement brand.

Target audience : {audience}
Section         : {section_name} — {section_desc}
Products        : {products}

════════════════════════════════════════
TASK
════════════════════════════════════════
Generate exactly {amount} original blog post idea(s).
Each must be unique, highly specific, and written for a real woman to search on Google.

════════════════════════════════════════
CONTENT QUALITY RULES
════════════════════════════════════════
TITLES:
- Must be phrased as a question a real woman would type into Google.
  GOOD: "Why Does Perimenopause Cause Night Sweats — And What Actually Helps?"
  BAD:  "Managing Hot Flashes During Menopause"
- Minimum 8 words.
- Must include a specific symptom, mechanism, or outcome.
- Must NOT be similar to any existing title below.

KEYWORDS:
- Primary: one specific long-tail keyword (3–5 words) with clear search intent.
- Secondary: two related supporting keywords.

SUMMARIES:
- One sentence stating the article's core angle and who it helps.
- Must reference a specific biological mechanism or symptom.

STRICT EXCLUSIONS:
- No disease claims (treat/cure/prevent).
- No competitor references: {competitors}.
- No generic wellness topics with no menopause angle.

════════════════════════════════════════
EXISTING TITLES (do NOT repeat or closely resemble):
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
    def plan_next_posts(self, sections_config, existing_titles,
                        config=None, posts_per_section=1):
        """
        Main entry point called by the Orchestrator.
        Returns: {"status": str, "new_rows": [list of dicts],
                  "warnings": [], "errors": []}
        """
        from datetime import date

        cfg      = config or self.config
        errors   = []
        warnings = []
        new_rows = []

        # ── 1. VALIDATE INPUTS ──────────────────────────────────────────
        validation = validate_planner_inputs(sections_config, existing_titles, cfg)
        if not validation["valid"]:
            return {"status": "BLOCKED", "new_rows": [],
                    "warnings": validation["warnings"],
                    "errors":   validation["errors"]}
        warnings.extend(validation["warnings"])

        existing_lower = [t.lower() for t in existing_titles]
        today          = date.today()

        # ── 2. LOOP SECTIONS ────────────────────────────────────────────
        for section in sections_config:
            section_name = section.get("Name",        "")
            section_desc = section.get("Description", "")

            # _build_brainstorm_prompt signature:
            # (self, section_name, section_desc, existing_titles, amount, config)
            prompt = self._build_brainstorm_prompt(
                section_name    = section_name,
                section_desc    = section_desc,
                existing_titles = existing_titles,
                amount          = posts_per_section * 5,  # ask for extras as fallback
                config          = cfg
            )

            # ── 3. CALL GEMINI ───────────────────────────────────────────
            try:
                response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
                raw_text = response.text.strip()
            except Exception as e:
                errors.append(f"Gemini error for '{section_name}': {e}")
                continue

            # ── 4. PARSE JSON ARRAY FROM RESPONSE ───────────────────────
            # Your prompt returns a ```json [...] ``` block
            try:
                json_match = re.search(r"```json\s*(\[.*?\])\s*```",
                                       raw_text, re.DOTALL)
                if json_match:
                    ideas = json.loads(json_match.group(1))
                else:
                    # Fallback: try parsing the whole response as JSON
                    ideas = json.loads(raw_text)
            except (json.JSONDecodeError, AttributeError) as e:
                errors.append(
                    f"JSON parse failed for '{section_name}': {e}\n"
                    f"Raw output: {raw_text[:300]}"
                )
                continue

            # ── 5. VALIDATE + PICK IDEAS ─────────────────────────────────
            chosen_count = 0
            for idea in ideas:
                if chosen_count >= posts_per_section:
                    break

                # validate_topic signature: (idea_dict, existing_titles_lower)
                check = validate_topic(idea, existing_lower)
                if not check["valid"]:
                    warnings.append(
                        f"Skipped idea '{idea.get('Title','')}': {check['reason']}"
                    )
                    continue

                title   = idea.get("Title",   "").strip()
                keyword = idea.get("Keyword", "").strip()

                # ── 6. TREND CHECK ────────────────────────────────────────
                # check_keyword_popularity returns:
                # {"passed": bool, "score": float, "flagged": bool, "reason": str}
                trend = self.check_keyword_popularity(keyword)

                if not trend["passed"] and not trend["flagged"]:
                    warnings.append(
                        f"Low trend score for '{keyword}': {trend['reason']}"
                    )
                    # Don't skip — just flag it

                trend_score = trend["score"] if trend["score"] >= 0 else "N/A"
                trend_note  = "TrendFlagged" if trend["flagged"] else ""

                # ── 7. SCHEDULE DATE ──────────────────────────────────────
                scheduled_date = compute_scheduled_date(
                    base_date      = today,
                    post_index     = len(existing_titles) + len(new_rows),
                    posts_per_week = int(
                        cfg.get("cadence", {}).get("PostsPerWeek", 3)
                    )
                )

                # ── 8. BUILD ROW DICT ─────────────────────────────────────
                new_rows.append({
                    "Status":            "Pending Approval",
                    "Title":             title,
                    "Section":           section_name,
                    "Keyword":           keyword,
                    "SecondaryKeywords": idea.get("SecondaryKeywords", ""),
                    "Summary":           idea.get("Summary", ""),
                    "ScheduledDate":     scheduled_date,
                    "ScheduledTime":     "09:00 AM ET",
                    "ReviewerNotes":     trend_note,
                })

                # Prevent same-run duplicates
                existing_lower.append(title.lower())
                chosen_count += 1

            if chosen_count == 0:
                warnings.append(
                    f"Section '{section_name}': all {len(ideas)} ideas failed validation."
                )

        status = "BLOCKED" if (errors and not new_rows) else "SUCCESS"
        return {
            "status":   status,
            "new_rows": new_rows,
            "warnings": warnings,
            "errors":   errors,
        }
