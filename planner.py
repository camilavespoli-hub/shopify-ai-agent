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
# ══════════════════════════════════════════════════════════════════════════════


def validate_planner_inputs(sections_config, existing_titles, config):
    errors   = []
    warnings = []

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

    if not isinstance(existing_titles, list):
        errors.append("existing_titles must be a list (can be empty []).")

    brand = config.get("brand", {})

    if not brand.get("brand_name"):
        warnings.append("brand_name missing from Config_Brand — Gemini will use generic brand references.")
    if not brand.get("audience_age_range"):
        warnings.append("audience_age_range missing — Gemini may generate off-demographic topics.")
    if not brand.get("audience_gender"):
        warnings.append("audience_gender missing — audience targeting will be generic.")
    if not brand.get("audience_pain_points"):
        warnings.append("audience_pain_points missing — topic relevance may be reduced.")
    if not brand.get("competitor_names"):
        warnings.append("competitor_names missing — competitor topic avoidance will not be enforced.")
    if not brand.get("content_language"):
        warnings.append("content_language missing from Config_Brand — defaulting to English (en).")
    if not brand.get("industry"):
        warnings.append("industry missing from Config_Brand — compliance rules may not load correctly.")
    if not config.get("planner", {}).get("trend_score_threshold"):
        warnings.append("trend_score_threshold missing from Config_Planner — defaulting to 10.")

    # ── NEW: warn if topic_map or existing_records missing ───────────────────
    if not config.get("topic_map"):
        warnings.append(
            "topic_map not found in config — blog_type balance disabled. "
            "Add Config_TopicMap tab and pass it via orchestrator."
        )
    if not config.get("existing_records"):
        warnings.append(
            "existing_records not found in config — coverage gap will default to zero. "
            "Pass plan_sheet.get_all_records() via orchestrator."
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}



# ══════════════════════════════════════════════════════════════════════════════
# TOPIC VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════


def validate_topic(idea, existing_titles_lower):
    title   = idea.get("Title",   "").strip()
    keyword = idea.get("Keyword", "").strip()
    summary = idea.get("Summary", "").strip()

    if not title:
        return {"valid": False, "reason": "Title is empty."}

    word_count = len(title.split())
    if word_count < 5 and "?" not in title:
        return {"valid": False, "reason": f"Title too vague ({word_count} words, no question): '{title}'"}

    if not keyword or len(keyword.strip()) < 2:
        return {"valid": False, "reason": "Keyword is too short or empty."}

    if not summary or len(summary.strip()) < 10:
        return {"valid": False, "reason": "Summary is too short or empty."}

    title_lower = title.lower()
    clean_title = re.sub(r'[^\w\s]', '', title_lower)
    title_words = set(clean_title.split())

    if not title_words:
        return {"valid": False, "reason": "Title contains no valid words after cleaning."}

    for existing in existing_titles_lower:
        clean_existing = re.sub(r'[^\w\s]', '', existing)
        existing_words = set(clean_existing.split())
        if not existing_words:
            continue
        overlap = len(title_words & existing_words) / len(existing_words)
        if overlap > 0.6:
            return {"valid": False, "reason": f"Title too similar to existing: '{existing}'"}

    return {"valid": True, "reason": ""}



# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULING HELPERS
# ══════════════════════════════════════════════════════════════════════════════


DAY_NAME_TO_NUMBER = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}


def parse_publish_days(cadence_config):
    raw = cadence_config.get("publish_days", "Monday, Wednesday, Friday")
    day_numbers = []
    for day in raw.split(","):
        day_clean = day.strip().lower()
        if day_clean in DAY_NAME_TO_NUMBER:
            day_numbers.append(DAY_NAME_TO_NUMBER[day_clean])
        else:
            print(f"   ⚠️  Unknown publish day '{day.strip()}' — skipping.")
    return sorted(day_numbers) if day_numbers else [0, 2, 4]


def compute_scheduled_date(base_date, post_index, cadence_config, occupied_dates=None):
    """
    Asigna el próximo día disponible según publish_days,
    evitando fechas ya ocupadas por otros posts.
    """
    publish_days = parse_publish_days(cadence_config)
    raw_blackout = cadence_config.get("blackout_dates", "")
    blackout_set = set(d.strip() for d in raw_blackout.split(",") if d.strip())
    occupied_set = set(occupied_dates or [])
    days_between = int(cadence_config.get("days_between_posts", 1))

    # Genera candidatos de fechas hacia adelante
    candidate = base_date
    slots_found = 0
    max_days = 365  # safety limit

    for _ in range(max_days):
        if (candidate.weekday() in publish_days
                and candidate.strftime("%Y-%m-%d") not in blackout_set
                and candidate.strftime("%Y-%m-%d") not in occupied_set):

            if slots_found == post_index:
                return candidate.strftime("%Y-%m-%d")
            slots_found += 1
            occupied_set.add(candidate.strftime("%Y-%m-%d"))  # marca como ocupado

            for gap in range(1, days_between):
                blocked = candidate + timedelta(days=gap)
                occupied_set.add(blocked.strftime("%Y-%m-%d"))

        candidate += timedelta(days=1)

    return base_date.strftime("%Y-%m-%d")  # fallback




# ══════════════════════════════════════════════════════════════════════════════
# COVERAGE HELPER
# ══════════════════════════════════════════════════════════════════════════════


def compute_coverage_gap(existing_records, topic_map, config):
    # ── NEW: early return with warning if no topic_map ────────────────────────
    if not topic_map:
        print("   ⚠️  TopicMap is empty — blog_type balance disabled, defaulting to 'educational'.")
        return []

    system      = config.get("system", {})
    reset_every = int(system.get("reset_every_n_posts", 20))

    published = [
        r for r in existing_records
        if str(r.get("Published_Status", "")).strip().lower() == "published"
    ]

    cycle_position  = len(published) % reset_every
    cycle_start_idx = len(published) - cycle_position
    cycle_records   = published[cycle_start_idx:]

    current_counts = {}
    for r in cycle_records:
        key = (r.get("Section", "").strip(), r.get("BlogType", "").strip())
        current_counts[key] = current_counts.get(key, 0) + 1

    in_pipeline = [
        r for r in existing_records
        if str(r.get("Published_Status", "")).strip().lower() != "published"
        and str(r.get("Status", "")).strip().lower() not in ("", "rejected")
    ]
    pipeline_counts = {}
    for r in in_pipeline:
        key = (r.get("Section", "").strip(), r.get("BlogType", "").strip())
        pipeline_counts[key] = pipeline_counts.get(key, 0) + 1

    gaps = []
    for entry in topic_map:
        section   = entry.get("section",   "").strip()
        blog_type = entry.get("blog_type", "").strip()
        weight    = float(entry.get("monthly_weight", 0)) / 100
        cycle_goal = max(1, round(weight * reset_every))

        key       = (section, blog_type)
        current   = current_counts.get(key, 0) + pipeline_counts.get(key, 0)
        remaining = cycle_goal - current

        if remaining > 1:
            priority = "HIGH"
        elif remaining == 1:
            priority = "NORMAL"
        else:
            priority = "SATURATED"

        gaps.append({
            "section":   section,
            "blog_type": blog_type,
            "goal":      cycle_goal,
            "current":   current,
            "remaining": remaining,
            "priority":  priority,
        })

    # ── NEW: print coverage table for visibility ──────────────────────────────
    print("   📊 Coverage gaps (current cycle):")
    for g in sorted(gaps, key=lambda x: (x["section"], -x["remaining"])):
        icon = "⚠️ " if g["priority"] == "HIGH" else ("✅" if g["priority"] == "SATURATED" else "→ ")
        print(f"      {icon} [{g['section']:25s}] {g['blog_type']:20s} "
              f"{g['current']}/{g['goal']} — {g['priority']}")

    return gaps



# ══════════════════════════════════════════════════════════════════════════════
# PLANNER AGENT
# ══════════════════════════════════════════════════════════════════════════════


class PlannerAgent:

    def __init__(self, config=None):
        print("🧠 Initializing Planner Agent...")

        self.config = config or {"brand": {}, "planner": {}, "system": {}}

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("   ❌ GEMINI_API_KEY missing — Planner will fail.")

        self.client = genai.Client(api_key=api_key)

        self.model_name = (
            self.config.get("system", {}).get("Gemini_Model")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        print(f"   🤖 Model: {self.model_name}")

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
    # ──────────────────────────────────────────────────────────────────────────

    def check_keyword_popularity(self, keyword, retries=3):
        threshold = int(self.config.get("planner", {}).get("trend_score_threshold", 10))

        if not self.pytrends:
            return {"passed": True, "score": -1, "flagged": True,
                    "reason": "Pytrends unavailable — trend score skipped."}

        target_country = self.config.get("brand", {}).get("target_country", "US")
        print(f"   📈 Checking Google Trends: '{keyword}' ({target_country})...")

        for attempt in range(retries):
            try:
                self.pytrends.build_payload(
                    [keyword], cat=0, timeframe="today 12-m",
                    geo=target_country, gprop=""
                )
                data = self.pytrends.interest_over_time()
                time.sleep(5 + (attempt * 2))

                if data.empty or keyword not in data.columns:
                    return {"passed": False, "score": 0, "flagged": False,
                            "reason": f"No trend data found for '{keyword}'."}

                avg_score = round(float(data[keyword].mean()), 2)
                passed    = avg_score >= threshold
                print(f"      Score: {avg_score:.1f} (threshold: {threshold}) → "
                      f"{'✅ PASS' if passed else '❌ LOW'}")
                return {
                    "passed":  passed,
                    "score":   avg_score,
                    "flagged": False,
                    "reason":  "" if passed else f"Score {avg_score} is below threshold {threshold}."
                }

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "TooManyRequests" in error_msg.replace(" ", ""):
                    print("   🚨 Pytrends 429 rate limit hit — disabling trend checks for this run.")
                    self.pytrends = None
                    return {"passed": True, "score": -1, "flagged": True,
                            "reason": "Pytrends 429 rate limit — requires manual verification."}
                wait = 2 ** attempt * 10
                print(f"   ⚠️  Pytrends attempt {attempt + 1}/{retries} failed: {e}. "
                      f"Retrying in {wait}s...")
                time.sleep(wait)

        print(f"   ⚠️  Pytrends failed after {retries} retries. Flagging for review.")
        return {"passed": True, "score": -1, "flagged": True,
                "reason": f"Pytrends failed after {retries} retries — manual verification needed."}


    # ──────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ──────────────────────────────────────────────────────────────────────────

    def _build_brainstorm_prompt(self, section_name, section_desc,
                                  existing_titles, amount, config,
                                  target_blog_type="educational",
                                  coverage_context=""):

        brand = config.get("brand", {})

        brand_name       = str(brand.get("brand_name",              "the brand"))
        age_range        = str(brand.get("audience_age_range",      "38–55"))
        gender           = str(brand.get("audience_gender",         "female"))
        pain_points      = str(brand.get("audience_pain_points",    "fatigue, brain fog, weight gain"))
        sophistication   = str(brand.get("audience_sophistication", "educated, reads labels"))
        tone             = str(brand.get("tone_formality",          "conversational"))
        competitors      = str(brand.get("competitor_names",        ""))
        products         = str(brand.get("product_names",           ""))
        avoid_topics     = str(brand.get("avoid_topics",            "competitor names, political topics"))
        language         = str(brand.get("content_language",        "en"))
        industry         = str(brand.get("industry",                "supplements"))
        compliance       = str(brand.get("compliance_framework",    "FDA"))
        disclaimer       = str(brand.get("disclaimer_text",         ""))
        section_name     = str(section_name)
        section_desc     = str(section_desc)
        coverage_context = str(coverage_context)

        language_names = {
            "en": "English", "es": "Spanish", "pt": "Portuguese",
            "fr": "French",  "de": "German",  "it": "Italian"
        }
        language_label = language_names.get(language.lower(), language)

        titles_block = (
            "\n".join([f"- {t}" for t in existing_titles])
            if existing_titles else "None — this section is new."
        )

        compliance_rules = (
            f"- Industry: {industry}\n"
            f"- Regulatory framework: {compliance}\n"
            f"- Never make disease claims (cure / treat / fix / reverse / prevent).\n"
            f'- Always use hedging language: "may support", "research suggests", "can help".\n'
            f'- Append disclaimer where appropriate: "{disclaimer}"'
        )

        blog_types   = config.get("blog_types", {})
        bt_config    = blog_types.get(target_blog_type, {})
        includes_faq = bt_config.get("includes_faq", True)
        has_product  = bt_config.get("target_product_link", False)

        blog_type_instruction = (
            f"This article must include a soft CTA linking to one of the brand's "
            f"products: {products}. The CTA must feel helpful and non-salesy — "
            f"the article earns the recommendation."
            if has_product else ""
        )

        faq_instruction = (
            "Include a FAQ section at the end with 3–5 questions DIRECTLY related "
            "to the article title and keyword. Do NOT generate generic questions — "
            "each must be specific to THIS article's topic."
            if includes_faq else
            "Do NOT include a FAQ section in this article."
        )

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
CONTENT QUOTA STATUS (current cycle)
════════════════════════════════════════
{coverage_context if coverage_context else "No coverage data available — generate freely."}



════════════════════════════════════════
TASK
════════════════════════════════════════
Generate exactly {amount} original blog post idea(s).
Blog type for this batch: {target_blog_type.upper()}


Blog type behavior:
- educational    → Explains a symptom, mechanism, or concept. Informational intent.
- how-to guide   → Step-by-step actionable advice the reader can apply today.
- buying guide   → Helps the reader choose the right product. Soft CTA included.
- customer story → Real-feeling narrative of a woman's transformation. First-person tone.
- case study     → Data or outcome-driven. Focuses on results and mechanisms.


{blog_type_instruction}


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


FAQ INSTRUCTION:
{faq_instruction}


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
    "Summary": "Explains how fluctuating estrogen levels dysregulate the hypothalamus and trigger night sweats in perimenopausal women.",
    "BlogType": "{target_blog_type}",
    "TopicCluster": "night sweats"
  }}
]
```"""


    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def plan_next_posts(self, sections_config, existing_titles,
                        config=None, posts_per_section=1):
        cfg      = config or self.config
        errors   = []
        warnings = []
        new_rows = []

        # ── Step 1: Preflight validation ──────────────────────────────────────
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

        # ── Step 2: Coverage gap ──────────────────────────────────────────────
        topic_map        = cfg.get("topic_map",        [])
        existing_records = cfg.get("existing_records", [])
        coverage_gaps    = compute_coverage_gap(existing_records, topic_map, cfg)
        occupied_dates_runtime = list(cfg.get("occupied_dates", []))

        def _coverage_summary(section_name):
            section_gaps = [g for g in coverage_gaps if g["section"] == section_name]
            if not section_gaps:
                return ""
            lines = []
            for g in sorted(section_gaps, key=lambda x: -x["remaining"]):
                icon = "⚠️" if g["priority"] == "HIGH" else (
                       "✅" if g["priority"] == "SATURATED" else "→")
                status_label = (
                    "SATURATED" if g["priority"] == "SATURATED"
                    else f"need {g['remaining']} more"
                )
                lines.append(
                    f"  {icon} {g['blog_type']:20s}: {g['current']}/{g['goal']} ({status_label})"
                )
            return "\n".join(lines)

        def _pick_blog_type(section_name):
            section_gaps = [
                g for g in coverage_gaps
                if g["section"] == section_name and g["priority"] != "SATURATED"
            ]
            if not section_gaps:
                # ── NEW: log when falling back to educational ─────────────────
                print(f"   ℹ️  [{section_name}] All blog types SATURATED or no TopicMap "
                      f"— defaulting to 'educational'.")
                return "educational"
            top    = max(section_gaps, key=lambda x: x["remaining"])
            chosen = top["blog_type"]
            print(f"   🎯 [{section_name}] Selected blog_type: '{chosen}' "
                  f"(remaining: {top['remaining']}, priority: {top['priority']})")
            return chosen

        # ── Step 3: Loop through sections ─────────────────────────────────────
        for section in sections_config:
            section_name = section.get("Name",        "")
            section_desc = section.get("Description", "")

            target_blog_type = _pick_blog_type(section_name)
            coverage_context = _coverage_summary(section_name)
            request_amount   = posts_per_section * 5

            prompt = self._build_brainstorm_prompt(
                section_name     = section_name,
                section_desc     = section_desc,
                existing_titles  = existing_titles,
                amount           = request_amount,
                config           = cfg,
                target_blog_type = target_blog_type,
                coverage_context = coverage_context,
            )

            # ── Step 4: Call Gemini ───────────────────────────────────────────
            try:
                response = self.client.models.generate_content(
                    model    = self.model_name,
                    contents = [{"role": "user", "parts": [{"text": str(prompt)}]}]
                )
                raw_text = response.text.strip()
            except Exception as e:
                errors.append(f"Gemini error for section '{section_name}': {e}")
                continue

            # ── Step 5: Parse JSON response ───────────────────────────────────
            try:
                json_match = re.search(r"```json\s*(\[.*?\])\s*```",
                                       raw_text, re.DOTALL)
                if json_match:
                    ideas = json.loads(json_match.group(1))
                else:
                    ideas = json.loads(raw_text)
            except (json.JSONDecodeError, AttributeError) as e:
                errors.append(
                    f"JSON parse failed for section '{section_name}': {e}\n"
                    f"Raw output (first 300 chars): {raw_text[:300]}"
                )
                continue

            # ── Step 6: Validate and select ideas ────────────────────────────
            chosen_count = 0

            for idea in ideas:
                if chosen_count >= posts_per_section:
                    break

                check = validate_topic(idea, existing_lower)
                if not check["valid"]:
                    warnings.append(
                        f"Skipped idea '{idea.get('Title', '')}': {check['reason']}"
                    )
                    continue

                title   = idea.get("Title",   "").strip()
                keyword = idea.get("Keyword", "").strip()

                # ── Step 7: Trend check ───────────────────────────────────────
                trend = self.check_keyword_popularity(keyword)
                if not trend["passed"] and not trend["flagged"]:
                    warnings.append(f"Low trend score for '{keyword}': {trend['reason']}")

                trend_note = "TrendFlagged" if trend["flagged"] else ""

                # ── Step 8: Schedule date ─────────────────────────────────────
                scheduled_date = compute_scheduled_date(
                    base_date      = today,
                    post_index     = 0,  # siempre 0, runtime_list excluye los usados
                    cadence_config = cadence_config,
                    occupied_dates = occupied_dates_runtime
                )
                
                occupied_dates_runtime.append(scheduled_date)
                scheduled_time = cadence_config.get("publish_time", "09:00 AM ET")

                # ── Step 9: Build row dict ────────────────────────────────────
                raw_secondary = idea.get("SecondaryKeywords", "")
                secondary_keywords = (
                    ", ".join(raw_secondary)
                    if isinstance(raw_secondary, list)
                    else str(raw_secondary)
                )

                # ── NEW: enforce blog_type from picker, not Gemini's suggestion
                # Gemini may hallucinate a different blog_type — we override it
                final_blog_type = target_blog_type

                new_rows.append({
                    "Status":            "Pending Approval",
                    "Title":             title,
                    "Section":           section_name,
                    "Keyword":           keyword,
                    "SecondaryKeywords": secondary_keywords,
                    "Summary":           str(idea.get("Summary",      "")),
                    "ScheduledDate":     scheduled_date,
                    "ScheduledTime":     scheduled_time,
                    "ReviewerNotes":     trend_note,
                    "BlogType":          final_blog_type,   # ← always from picker
                    "TopicCluster":      str(idea.get("TopicCluster", "")),
                })

                existing_lower.append(title.lower())
                chosen_count += 1

            if chosen_count == 0:
                warnings.append(
                    f"Section '{section_name}': all {len(ideas)} generated ideas "
                    "failed validation — consider broadening the section description."
                )

        # ── Final result ──────────────────────────────────────────────────────
        status = "BLOCKED" if (errors and not new_rows) else "SUCCESS"

        return {
            "status":   status,
            "new_rows": new_rows,
            "warnings": warnings,
            "errors":   errors,
        }
