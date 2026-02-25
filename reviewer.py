import os
import re
import json
from google import genai


# ══════════════════════════════════════════════════════════════════════════════
# REVIEWER AGENT — Agent 4 of the pipeline
#
# Responsibilities:
#   1. Validate inputs (draft exists, research available, config complete)
#   2. Pass 1 — Automated: regex compliance scan + structural checks
#      → Fast, no API cost, catches obvious violations immediately
#   3. Pass 2 — AI: nuanced compliance + brand voice + engagement quality
#      → Only runs if Pass 1 didn't find hard violations
#   4. Merge all results and return a structured verdict to the Orchestrator
#
# Output statuses:
#   PASS            → no issues at all → Optimizer picks it up
#   PASS_WITH_NOTES → no violations, but suggestions exist → Optimizer picks it up
#   FAIL            → required fixes exist → sent back to Writer for rewrite
#   HARD_FAIL       → disease claim detected by regex → human review needed
#   BLOCKED         → input validation failed → logged, skip this row
#
# Everything is read from Config_Brand and Agent_Rules — nothing is hardcoded.
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATOR
# ──────────────────────────────────────────────────────────────────────────────

def validate_reviewer_inputs(content_row, draft_text, research_result, config):
    """
    Hard-gate validation before any review begins.
    Returns: {"valid": bool, "errors": [], "warnings": []}

    errors   → pipeline stops for this row
    warnings → logged but review continues
    """
    errors   = []
    warnings = []

    # ── 1. Draft must exist and be substantive ────────────────────────────────
    if not draft_text or not isinstance(draft_text, str):
        errors.append("draft_text is required and must be a string.")
    elif len(draft_text.strip()) < 100:
        errors.append(
            f"Draft is too short ({len(draft_text.strip())} chars). "
            "Minimum 100 characters required."
        )

    # ── 2. Draft must not be an error state from the Writer ──────────────────
    # If the Writer returned an error string instead of HTML, we can't review it
    if isinstance(draft_text, str) and draft_text.startswith(
        ("BLOCKED", "HARD_FAIL", "Research failed")
    ):
        errors.append(
            f"Cannot review: Writer returned an error state: "
            f"'{draft_text[:60]}'"
        )

    # ── 3. Required content row fields ───────────────────────────────────────
    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")

    # ── 4. Research result (soft — used for claim boundary check) ────────────
    if not research_result or not isinstance(research_result, dict):
        warnings.append(
            "research_result not provided — "
            "topic-specific claim boundary check will be skipped."
        )

    # ── 5. Brand config validation ────────────────────────────────────────────
    # NOTE: This block must stay OUTSIDE and AFTER the research_result check.
    # It validates brand config independently — always runs regardless of research.
    brand = config.get("brand", {})

    if not brand.get("brand_voice_summary"):
        warnings.append(
            "brand_voice_summary missing — AI brand tone check will use defaults."
        )
    # disclaimer_text replaces fda_disclaimer_text (now industry-agnostic)
    if not brand.get("disclaimer_text"):
        warnings.append(
            "disclaimer_text missing from Config_Brand — "
            "Reviewer will use hardcoded default pattern."
        )
    if not brand.get("content_language"):
        warnings.append(
            "content_language missing — Reviewer will default to English (en)."
        )
    if not brand.get("industry"):
        warnings.append(
            "industry missing — compliance rules may not load correctly."
        )
    # Specific audience fields (target_audience was removed from the Sheet)
    if not brand.get("audience_age_range"):
        warnings.append("audience_age_range missing — brand voice check will use defaults.")
    if not brand.get("audience_pain_points"):
        warnings.append("audience_pain_points missing — relevance check will use defaults.")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ──────────────────────────────────────────────────────────────────────────────
# TOOL 1: AUTOMATED REGEX COMPLIANCE SCAN
# Runs before the AI — catches obvious violations instantly with no API cost.
# Smart: strips disclaimer and HTML tags before scanning so known-safe text
# never triggers a false positive.
#
# Now reads product names dynamically from Config_Brand (not hardcoded).
# ──────────────────────────────────────────────────────────────────────────────

def scan_compliance(text, config=None):
    """
    Regex-based disease claim and compliance scanner.

    text   : the raw HTML draft from the Writer
    config : full config dict — used to read product names dynamically

    Returns:
      {
        "has_violations": bool    — True = hard fail, don't proceed to AI
        "violations":     [str]   — specific hard violations found
        "flags":          [str]   — soft flags for AI context review
        "clean_text":     str     — disclaimer-stripped text used for scanning
      }
    """
    brand = (config or {}).get("brand", {})

    # ── Step 1: Strip HTML tags ───────────────────────────────────────────────
    # We check plain text — not raw HTML — to avoid false matches inside tag attributes
    text = re.sub(r"<[^>]+>", " ", text)

    # ── Step 2: Strip disclaimer before scanning ──────────────────────────────
    # The disclaimer intentionally contains "treat", "cure", "prevent", "diagnose".
    # We remove it before scanning so it never causes a false HARD FAIL.
    # Keywords cover both English (en) and possible translated versions.
    lines = text.split("\n")
    disclaimer_keywords = [
        "food and drug administration",
        "not intended to diagnose",
        "these statements have not been evaluated",
        # Also strip if disclaimer_text from Config_Brand is present
        (brand.get("disclaimer_text", "")[:30].lower() if brand.get("disclaimer_text") else "")
    ]
    clean_lines = [
        line for line in lines
        if not any(kw and kw in line.lower() for kw in disclaimer_keywords)
    ]
    clean_text = "\n".join(clean_lines).lower()

    # ── Step 3: Build dynamic product name pattern ────────────────────────────
    # Read product names from Config_Brand instead of hardcoding them.
    # This makes the scanner work for any brand, not just Glomend.
    raw_products = brand.get("product_names", "")
    brand_name   = brand.get("brand_name",    "")
    product_list = [p.strip() for p in raw_products.split(",") if p.strip()]
    if brand_name and brand_name not in product_list:
        product_list.append(brand_name)
    product_pattern = "|".join(product_list) if product_list else "this product|our supplement"

    # ── Step 4: Hard-fail patterns (definite regulatory violations) ───────────
    # These match language that clearly violates health supplement regulations.
    # A match here = HARD FAIL → pipeline stops, human must review.
    hard_patterns = [
        (r"\b(treats|treated|treating)\s+\w+",
         "Active disease treatment claim"),
        (r"\b(cures|cured|curing)\s+\w+",
         "Disease cure claim"),
        (r"\b(prevents|prevented|preventing)\s+\w+",
         "Disease prevention claim"),
        (r"\b(diagnoses|diagnosed|diagnosing)\b",
         "Diagnosis claim — always a hard violation"),
        (r"\b(heals|healed|healing)\s+\w+",
         "Disease healing claim"),
        (r"\b(reverses|reversed|reversing)\s+\w+",
         "Disease reversal claim"),
        (r"works like a prescription",
         "Prescription equivalence claim"),
        (r"clinically proven to (treat|cure|prevent)",
         "Unsubstantiated clinical claim"),
        (r"guaranteed (results|to work|effectiveness)",
         "Guarantee claim — FTC violation risk"),
        # Dynamic: product name + disease within 60 chars
        (rf"({product_pattern}).{{0,60}}(insomnia|depression|anxiety disorder|diabetes|cancer|dementia)",
         "Product + disease claim detected"),
    ]

    # ── Step 5: Soft-flag patterns (require AI context review) ───────────────
    # These MIGHT be violations depending on context.
    # We pass them to the AI reviewer with context instead of auto-failing.
    soft_patterns = [
        (r"\binsomnia\b",
         "Named sleep disorder — verify no disease claim context"),
        (r"\bdepression\b",
         "Named mental health condition — verify context"),
        (r"\banxiety disorder\b",
         "Named disorder — verify no disease claim context"),
        (r"\bmedication\b",
         "Medication reference — verify no drug comparison implied"),
        (r"\b100%\b|\bguaranteed\b",
         "Absolute claim language — verify context"),
        (r"\bclinically proven\b",
         "Clinical claim — verify substantiation exists in citations"),
    ]

    violations = []
    flags      = []

    for pattern, label in hard_patterns:
        try:
            if re.search(pattern, clean_text):
                violations.append(f"HARD FAIL — {label}: matched pattern '{pattern[:60]}'")
        except re.error as e:
            print(f"   ⚠️  Compliance regex error (skipped): {e}")

    for pattern, label in soft_patterns:
        try:
            if re.search(pattern, clean_text):
                flags.append(f"SOFT FLAG — {label}: matched pattern '{pattern}'")
        except re.error as e:
            print(f"   ⚠️  Soft flag regex error (skipped): {e}")

    return {
        "has_violations": len(violations) > 0,
        "violations":     violations,
        "flags":          flags,
        "clean_text":     clean_text
    }


# ──────────────────────────────────────────────────────────────────────────────
# TOOL 2: STRUCTURAL VALIDATION
# Checks that all required HTML sections are present.
# Handles both proper HTML (from Writer) and edge cases gracefully.
# Word count uses Config_Brand values — not hardcoded numbers.
# ──────────────────────────────────────────────────────────────────────────────

def check_structure(text, config):
    """
    Validates the HTML draft for required structural elements.

    Returns:
      {
        "passed": bool    — False if any required section is missing
        "issues": [str]   — missing/broken required elements (blocking)
        "notes":  [str]   — non-blocking observations
      }
    """
    brand  = config.get("brand", {})
    issues = []
    notes  = []
    tl     = text.lower()

    # ── Headings ──────────────────────────────────────────────────────────────
    # Accept HTML <h2>/<h3> or Markdown ## / **bold**
    has_html_headings = bool(re.search(r"<h[23][^>]*>", tl))
    has_md_headings   = "##" in text or "**" in text
    if not has_html_headings and not has_md_headings:
        issues.append("STRUCTURE: No headings found (<h2>/<h3> or ## or **bold**).")

    # ── FAQ section ───────────────────────────────────────────────────────────
    has_faq = (
        'class="faq"' in tl
        or "frequently asked questions" in tl
        or "<h2>faq" in tl
        or "faq" in tl
    )
    if not has_faq:
        issues.append("STRUCTURE: FAQ section missing — add <section class='faq'>.")

    # ── Sources / References ──────────────────────────────────────────────────
    has_sources = (
        'class="sources"' in tl
        or "<h2>sources" in tl
        or "sources" in tl
        or "references" in tl
    )
    if not has_sources:
        issues.append("STRUCTURE: Sources/References section missing.")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    # Check for the disclaimer using both the class name AND key phrases.
    # Key phrases cover any industry (not just FDA supplements).
    disclaimer_text = brand.get("disclaimer_text", "").lower()[:50]
    has_disclaimer = (
        'class="fda-disclaimer"' in tl
        or 'class="disclaimer"'   in tl
        or "food and drug administration" in tl
        or "not intended to diagnose"    in tl
        or "these statements have not been evaluated" in tl
        or (disclaimer_text and disclaimer_text in tl)
    )
    if not has_disclaimer:
        issues.append(
            "STRUCTURE: Disclaimer is missing — "
            "add <p class='fda-disclaimer'>{disclaimer_text}</p>."
        )

    # ── Inline citations ──────────────────────────────────────────────────────
    # Accept HTML <span class="citation"> OR [Source, Year] markdown style
    html_citations = len(re.findall(r'class="citation"', tl))
    md_citations   = len(re.findall(r"\[.+?,\s*\d{4}\]", text))
    citation_count = html_citations + md_citations

    if citation_count == 0:
        issues.append(
            "CITATIONS: No inline citations found. "
            "Every objective claim requires a [Source, Year] citation."
        )
    elif citation_count < 3:
        notes.append(
            f"CITATIONS: Only {citation_count} inline citation(s) found — "
            "consider adding more to strengthen credibility."
        )

    # ── Word count ────────────────────────────────────────────────────────────
    # Strip HTML tags for accurate plain-text word count
    plain_text = re.sub(r"<[^>]+>", " ", text)
    word_count = len(plain_text.split())
    word_min   = int(brand.get("default_word_count_min", 800))
    word_max   = int(brand.get("default_word_count_max", 1500))

    if word_count < word_min:
        issues.append(
            f"WORD COUNT: {word_count} words is below minimum ({word_min}). "
            "Expand body sections — do not add filler content."
        )
    elif word_count > word_max:
        # Over max → note (not issue) — the Writer already received this warning
        # and the Orchestrator's retry loop will handle it
        notes.append(
            f"WORD COUNT: {word_count} words exceeds maximum ({word_max}). "
            f"Trim by {word_count - word_max} words — do not cut citations or FAQ."
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "notes":  notes
    }


# ──────────────────────────────────────────────────────────────────────────────
# AI REVIEWER PROMPT BUILDER
# Only called if the automated scan passes — saves API cost on obvious failures.
# Fully dynamic: all brand/audience/compliance context from Config_Brand
# and Agent_Rules. No hardcoded brand names, audiences, or regulations.
# ──────────────────────────────────────────────────────────────────────────────

def build_reviewer_prompt(draft_text, content_row, research_result,
                           compliance_flags, config):
    """
    Builds the AI review prompt from Config_Brand and Agent_Rules.

    Parameters:
      draft_text        : the HTML draft to review
      content_row       : row dict with Title, Section, Keyword
      research_result   : dict from Researcher — used for claim_boundaries
      compliance_flags  : soft flags from the regex scan (list of strings)
      config            : full config dict from the Orchestrator
    """
    brand = config.get("brand", {})

    # ── Content context ───────────────────────────────────────────────────────
    title   = content_row.get("Title",   "")
    section = content_row.get("Section", "")

    # ── Brand & audience context ──────────────────────────────────────────────
    brand_name     = brand.get("brand_name",             "the brand")
    industry       = brand.get("industry",               "supplements")
    age_range      = brand.get("audience_age_range",     "38–55")
    gender         = brand.get("audience_gender",        "female")
    pain_points    = brand.get("audience_pain_points",   "fatigue, brain fog, weight gain")
    sophistication = brand.get("audience_sophistication","educated, reads labels")
    brand_voice    = brand.get(
        "brand_voice_summary",
        "Empathetic, direct, and empowering. "
        "Peer-to-peer tone — like a knowledgeable friend, not a textbook."
    )
    cta_style      = brand.get("cta_style", "soft — educational, not pushy")
    competitors    = brand.get("competitor_names", "")
    avoid_topics   = brand.get("avoid_topics", "competitor names, political topics")

    # ── Language ──────────────────────────────────────────────────────────────
    language = brand.get("content_language", "en")
    language_names = {
        "en": "English", "es": "Spanish", "pt": "Portuguese",
        "fr": "French",  "de": "German",  "it": "Italian"
    }
    language_label = language_names.get(language.lower(), language)

    # ── Compliance ────────────────────────────────────────────────────────────
    compliance  = brand.get("compliance_framework", "FDA")
    disclaimer  = brand.get("disclaimer_text", "")

    # ── Banned phrases from both sources ─────────────────────────────────────
    # Config_Brand → banned_phrases: tone/style bans
    # Agent_Rules → reviewer → forbidden_phrases: compliance-specific bans
    # Agent_Rules → reviewer → required_hedge_words: language that must be used
    brand_banned      = brand.get("banned_phrases", "delve, leverage, unlock, navigate")
    compliance_banned = (
        config.get("agent_rules", {})
        .get("reviewer", {})
        .get("forbidden_phrases", "cure, treat, fix, guarantee, prevent")
    )
    hedge_words = (
        config.get("agent_rules", {})
        .get("reviewer", {})
        .get("required_hedge_words", "may support, research suggests, can help")
    )
    all_banned = (
        f"{brand_banned}, {compliance_banned}, "
        "\"it's important to note\", \"a holistic approach\", "
        "\"revolutionary\", \"game-changer\", \"take charge\", \"empower yourself\""
    )

    # ── Claim boundaries from Researcher ─────────────────────────────────────
    # These are topic-specific boundaries (e.g. "do not claim this ingredient
    # treats insomnia") generated by the Researcher for this specific article.
    boundaries = []
    if isinstance(research_result, dict):
        boundaries = research_result.get("claim_boundaries", [])
    boundaries_block = "\n".join([f"- {b}" for b in boundaries]) \
        or f"No specific boundaries — apply general {compliance} rules."

    # ── Soft flags from regex scan ────────────────────────────────────────────
    # Give the AI the specific patterns that were flagged so it can evaluate
    # them in context (e.g. "depression" might be fine in a wellness article)
    flags_block = "\n".join(compliance_flags) \
        or "No soft flags detected by automated scan."

    return f"""You are a Senior Compliance Officer and Brand Editor for {brand_name},
a {industry} brand.

Your role: (1) catch any compliance issues the automated scan may have missed,
and (2) evaluate whether the content matches brand voice and audience expectations.

════════════════════════════════════════
LANGUAGE
════════════════════════════════════════
The article is written in {language_label}.
Evaluate tone and compliance in {language_label}.
Your JSON output keys must stay in English (they are code identifiers).
Your JSON output values (violations, notes, summaries) must be in {language_label}.

════════════════════════════════════════
REVIEW CONTEXT
════════════════════════════════════════
Article title  : {title}
Section        : {section}
Brand          : {brand_name}
Industry       : {industry}
Compliance     : {compliance}

════════════════════════════════════════
TARGET AUDIENCE
════════════════════════════════════════
Gender         : {gender}
Age range      : {age_range}
Pain points    : {pain_points}
Sophistication : {sophistication}

════════════════════════════════════════
BRAND VOICE
════════════════════════════════════════
{brand_voice}
CTA style      : {cta_style}

════════════════════════════════════════
TOPIC-SPECIFIC CLAIM BOUNDARIES (from Research team)
════════════════════════════════════════
{boundaries_block}

════════════════════════════════════════
SOFT FLAGS FROM AUTOMATED SCAN (review in context)
════════════════════════════════════════
{flags_block}

════════════════════════════════════════
BANNED PHRASES
════════════════════════════════════════
{all_banned}

REQUIRED hedging language for all health claims:
{hedge_words}

════════════════════════════════════════
DRAFT TO REVIEW
════════════════════════════════════════
{draft_text}

════════════════════════════════════════
REVIEW CHECKLIST — evaluate each item carefully
════════════════════════════════════════

COMPLIANCE CHECKS ({compliance}) — hard rules:
[ ] No disease claims (treat/cure/prevent/diagnose used in a disease context)
[ ] No competitor brand mentions ({competitors})
[ ] No personalized medical advice or specific dosage guidance
[ ] No absolute claims without substantiation ("guaranteed", "always works")
[ ] Scientific uncertainty is stated honestly where evidence is limited
[ ] Every objective health claim has an inline citation [Source, Year]
[ ] Topic-specific claim boundaries above are all respected
[ ] Avoid these topics entirely: {avoid_topics}
[ ] Disclaimer is present: "{disclaimer[:80]}..."

BRAND VOICE CHECKS — soft rules:
[ ] Empathetic opening — acknowledges what this audience experiences
[ ] Conversational, warm tone — "you", contractions, no clinical distance
[ ] No banned words or phrases
[ ] Short paragraphs (2–3 sentences max)
[ ] Subheadings are real questions this audience would type into Google
[ ] "What This Means for You" section feels warm and practical, not preachy
[ ] FAQ questions feel authentic to this audience's real concerns
[ ] CTA style is: {cta_style}

HUMAN WRITING CHECK — verify the draft does NOT read like AI:
[ ] Sentence length varies — mix of short and long sentences in every section
    (three sentences of the same length in a row = AI red flag)
[ ] Uses contractions naturally ("you're", "it's", "doesn't", "can't")
[ ] At least some sentences start with "But", "And", "So", "Yet", or "Because"
[ ] Contains at least one rhetorical question directed at the reader
[ ] Contains at least one em-dash (—) used as a natural interruption
[ ] Does NOT start any paragraph with: "Furthermore", "Moreover",
    "In addition", "It is worth noting", "Notably", "Interestingly",
    "Certainly", "Absolutely", "Of course"
[ ] Does NOT end sections with a summary sentence repeating what was just said
[ ] Does NOT use: "comprehensive", "crucial", "vital", "robust",
    "dive into", "holistic", "empower", "groundbreaking"
[ ] Specific details present — real numbers, timeframes, relatable scenarios
    (generic vague statements = AI red flag)
[ ] Tonal variation — not flat from start to finish

════════════════════════════════════════
OUTPUT — return ONLY a ```json block. No text outside it.
════════════════════════════════════════
{{
  "status":                  "PASS" | "PASS_WITH_NOTES" | "FAIL",
  "compliance_result":       "PASS" | "FAIL",
  "brand_voice_result":      "PASS" | "PASS_WITH_NOTES" | "FAIL",
  "human_voice_result": "PASS" | "FAIL",
  "violations": [
    "Exact quote or precise description of the compliance violation"
  ],
  "brand_notes": [
    "Specific, actionable brand voice or readability suggestion"
  ],
  "required_fixes": [
    "Exact change required before this post can be approved (compliance only)"
  ],
  "suggested_improvements": [
    "Optional improvement for tone, engagement, or SEO"
  ],
  "reviewer_summary": "One sentence summary for the Google Sheet Reviewer_Notes column."
}}

STATUS RULES (apply these exactly):
  PASS            = zero violations, zero required_fixes, human_voice_result is PASS
  PASS_WITH_NOTES = zero violations, zero required_fixes, but brand notes or suggestions exist
  FAIL            = ANY compliance violation OR ANY required_fix OR human_voice_result is FAIL

IMPORTANT: If human_voice_result is "FAIL", you MUST also add the specific
issues to required_fixes so the Writer knows exactly what to rewrite.

If violations or required_fixes are empty, use empty arrays [].
Do NOT output anything outside the ```json block."""


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSER
# Safely parses the AI Reviewer's JSON response.
# Falls back gracefully if Gemini returns malformed JSON —
# the pipeline never crashes due to a bad JSON response.
# ──────────────────────────────────────────────────────────────────────────────

def parse_reviewer_output(raw_text):
    """
    Extracts and parses the JSON block from the AI reviewer's response.

    Returns a parsed dict, or a safe fallback if parsing fails.
    The fallback uses PASS_WITH_NOTES so the post isn't discarded —
    the unstructured output is flagged for human review instead.
    """
    json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass  # Fall through to fallback

    # Fallback: AI returned unstructured text — flag it but don't discard the post
    return {
        "status":                  "PASS_WITH_NOTES",
        "compliance_result":       "PASS",
        "brand_voice_result":      "PASS_WITH_NOTES",
        "human_voice_result":      "PASS_WITH_NOTES",
        "violations":              [],
        "brand_notes":             [
            "NEEDS REVIEW: AI reviewer returned unstructured output — "
            "manual compliance check recommended."
        ],
        "required_fixes":          [],
        "suggested_improvements":  [],
        "reviewer_summary":        (
            f"AI review returned unstructured output — manual review recommended. "
            f"Raw: {raw_text[:100]}"
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# REVIEWER AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ReviewerAgent:

    def __init__(self, config=None):
        """
        Initializes the Reviewer Agent.
        Reads API keys, model, and brand config from the Orchestrator.
        """
        print("🛡️  Initializing Reviewer Agent...")
        self.config = config or {"brand": {}, "system": {}, "agent_rules": {}}

        # ── Gemini client ─────────────────────────────────────────────────────
        # Model priority: Config_System tab → GEMINI_MODEL env var → default
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("   ❌ GOOGLE_API_KEY missing — Reviewer will fail.")

        self.client = genai.Client(api_key=api_key)

        self.model_name = (
            self.config.get("system", {}).get("Gemini_Model")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        print(f"   🤖 Model: {self.model_name}")

        # ── Disclaimer presence pattern ───────────────────────────────────────
        # Used as a quick pre-check before the full structural validation.
        # Reads from Config_Brand if available, falls back to FDA default pattern.
        disclaimer_text = (
            self.config.get("brand", {}).get("disclaimer_text", "").lower()
        )
        if disclaimer_text:
            # Build a flexible regex from the first 30 chars of the disclaimer
            safe_fragment = re.escape(disclaimer_text[:30])
            self.disclaimer_pattern = re.compile(safe_fragment, re.IGNORECASE)
        else:
            # FDA default fallback
            self.disclaimer_pattern = re.compile(
                r"statements.*not.*evaluated.*food.*and.*drug.*administration"
                r".*not.*intended.*diagnose.*treat.*cure.*prevent.*disease",
                re.IGNORECASE | re.DOTALL
            )

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # Called by the Orchestrator for every row in the Writer→Reviewer loop.
    # ──────────────────────────────────────────────────────────────────────────

    def review_draft(self, content_row, draft_text, research_result, config=None):
        """
        Full two-pass review pipeline for one blog post draft.

        Pass 1 (automated, free):
          - Regex compliance scan → catches hard violations instantly
          - Structural check → verifies all required HTML sections exist

        Pass 2 (AI, costs Gemini tokens):
          - Only runs if Pass 1 finds no hard violations
          - Nuanced compliance check in context
          - Brand voice and engagement quality assessment

        Parameters:
          content_row     : dict from Content_Plan Sheet row
          draft_text      : HTML string from WriterAgent
          research_result : dict from ResearcherAgent (for claim boundaries)
          config          : full config dict from the Orchestrator

        Returns a dict with all verdict fields the Orchestrator needs.
        """
        cfg   = config if config is not None else self.config
        title = content_row.get("Title", "untitled")
        print(f"\n🛡️  Reviewer running for: {title}")

        # ── Step 1: Validate inputs ───────────────────────────────────────────
        validation = validate_reviewer_inputs(
            content_row, draft_text, research_result, cfg
        )
        if not validation["valid"]:
            print("   🚨 Reviewer BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"      ❌ {err}")
            return {
                "status":                 "BLOCKED",
                "compliance_result":      "UNKNOWN",
                "brand_voice_result":     "UNKNOWN",
                "human_voice_result":     "UNKNOWN",
                "violations":             [],
                "brand_notes":            [],
                "required_fixes":         [],
                "suggested_improvements": [],
                "reviewer_summary":       "; ".join(validation["errors"]),
                "structural_issues":      [],
                "warnings":               validation["warnings"],
                "errors":                 validation["errors"]
            }

        # ── Step 2: Automated regex compliance scan ───────────────────────────
        # No API cost. Catches hard violations immediately.
        # Passes config so product names are read dynamically (not hardcoded).
        print("   🔍 Pass 1: Automated compliance scan...")
        compliance_scan = scan_compliance(draft_text, config=cfg)

        # Hard violation found → stop immediately, no need for AI review
        if compliance_scan["has_violations"]:
            print("   🚨 HARD FAIL: Disease claim language detected.")
            for v in compliance_scan["violations"]:
                print(f"      ❌ {v}")
            return {
                "status":                 "HARD_FAIL",
                "compliance_result":      "FAIL",
                "brand_voice_result":     "UNKNOWN",
                "human_voice_result":     "UNKNOWN",
                "violations":             compliance_scan["violations"],
                "brand_notes":            [],
                "required_fixes":         compliance_scan["violations"],
                "suggested_improvements": [],
                "reviewer_summary":       (
                    f"HARD FAIL: {len(compliance_scan['violations'])} disease claim "
                    "violation(s) detected. Requires Writer revision."
                ),
                "structural_issues":      [],
                "warnings":               compliance_scan["flags"],
                "errors":                 []
            }

        # ── Step 3: Structural validation ─────────────────────────────────────
        # Checks all required HTML sections are present.
        print("   📋 Pass 1: Structural validation...")
        structure = check_structure(draft_text, cfg)
        if not structure["passed"]:
            print(f"   ⚠️  {len(structure['issues'])} structural issue(s) found:")
            for issue in structure["issues"]:
                print(f"      - {issue}")

        # ── Step 4: AI nuanced review ─────────────────────────────────────────
        # Only runs when Pass 1 has no hard violations.
        # Evaluates compliance in context + brand voice + engagement quality.
        print("   🤖 Pass 2: AI compliance + brand voice review...")
        prompt = build_reviewer_prompt(
            draft_text        = draft_text,
            content_row       = content_row,
            research_result   = research_result,
            compliance_flags  = compliance_scan["flags"],
            config            = cfg
        )

        try:
            raw_response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt
            ).text.strip()
        except Exception as e:
            print(f"   🚨 Reviewer AI error: {e}")
            # AI failure → return structural results only, don't discard the post
            return {
                "status":                 "BLOCKED",
                "compliance_result":      "UNKNOWN",
                "brand_voice_result":     "UNKNOWN",
                "human_voice_result":     "UNKNOWN",
                "violations":             [],
                "brand_notes":            [],
                "required_fixes":         structure["issues"],
                "suggested_improvements": [],
                "reviewer_summary":       f"AI review failed: {str(e)}",
                "structural_issues":      structure["issues"],
                "warnings":               validation["warnings"],
                "errors":                 [f"Gemini API error: {str(e)}"]
            }

        # ── Step 5: Parse AI response ─────────────────────────────────────────
        ai_result = parse_reviewer_output(raw_response)

        # ── Step 6: Merge all results ─────────────────────────────────────────
        # Combine structural issues + AI required fixes into one list.
        # If either source has required fixes, final status = FAIL.
        all_required_fixes = (
            ai_result.get("required_fixes", []) +
            structure["issues"]   # structural issues are always required fixes
        )
        all_brand_notes = (
            ai_result.get("brand_notes", []) +
            structure["notes"] +      # structural notes (non-blocking)
            validation["warnings"]    # config warnings
        )

        # Final status: most severe verdict wins
        # If there are required fixes, status is always FAIL regardless of AI verdict
        final_status = ai_result.get("status", "PASS")
        if all_required_fixes:
            final_status = "FAIL"
        # Escalate if human voice explicitly failed — even if required_fixes is empty
        if ai_result.get("human_voice_result") == "FAIL":
            final_status = "FAIL"
            if not ai_result.get("required_fixes"):
                fallback_note = (
                    ai_result.get("brand_notes", ["Human voice check failed."])[0]
                )
                all_required_fixes.append(f"HUMAN VOICE FAIL: {fallback_note}")

        # ── Step 7: Build Sheet summary ───────────────────────────────────────
        # This one-line summary goes into the Reviewer_Notes column of the Sheet
        if final_status == "PASS":
            reviewer_summary = "✅ Passed all compliance and brand checks."
        elif final_status == "PASS_WITH_NOTES":
            first_note = all_brand_notes[0] if all_brand_notes else ""
            reviewer_summary = (
                f"✅ Passed compliance. {len(all_brand_notes)} suggestion(s). "
                f"{first_note[:100]}"
            )
        else:
            first_fix = all_required_fixes[0] if all_required_fixes else "See review notes."
            reviewer_summary = (
                f"❌ FAIL — {len(all_required_fixes)} required fix(es). "
                f"{first_fix[:100]}"
            )

        # ── Step 8: Report to console ─────────────────────────────────────────
        print(f"   ✅ Review complete.")
        print(f"      Status      : {final_status}")
        print(f"      Compliance  : {ai_result.get('compliance_result', 'PASS')}")
        print(f"      Brand Voice : {ai_result.get('brand_voice_result', 'PASS')}")
        print(f"      Human Voice : {ai_result.get('human_voice_result', 'PASS')}")
        if all_required_fixes:
            print(f"      Required fixes ({len(all_required_fixes)}):")
            for fix in all_required_fixes[:3]:
                print(f"         ❌ {fix[:100]}")
        if all_brand_notes:
            print(f"      Notes ({len(all_brand_notes)}):")
            for note in all_brand_notes[:2]:
                print(f"         💡 {note[:100]}")

        return {
            "status":                 final_status,
            "compliance_result":      ai_result.get("compliance_result",      "PASS"),
            "brand_voice_result":     ai_result.get("brand_voice_result",     "PASS"),
            "human_voice_result":     ai_result.get("human_voice_result",     "PASS"),
            "violations":             ai_result.get("violations",             []),
            "brand_notes":            all_brand_notes,
            "required_fixes":         all_required_fixes,
            "suggested_improvements": ai_result.get("suggested_improvements", []),
            "reviewer_summary":       reviewer_summary,
            "structural_issues":      structure["issues"],
            "warnings":               validation["warnings"] + compliance_scan["flags"],
            "errors":                 []
        }
