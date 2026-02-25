import os
import re
import json
from google import genai
from bs4 import BeautifulSoup


# ══════════════════════════════════════════════════════════════════════════════
# WRITER AGENT — Agent 3 of the pipeline
#
# Responsibilities:
#   1. Validate inputs (research brief, brand config)
#   2. Extract and format research context for the prompt
#   3. Build a fully dynamic Gemini prompt from Config_Brand
#   4. Call Gemini to generate the HTML blog post draft
#   5. Run compliance check (no disease claims)
#   6. Parse + validate the output (structure, word count)
#   7. Return a structured result dict to the Orchestrator
#
# On retry: receives the previous draft + required_fixes from the Reviewer
# and rewrites completely — never copies the rejected draft.
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATOR
# Runs before any Gemini call to catch config/data problems early.
# Hard errors (errors[]) block the pipeline for this row.
# Soft issues (warnings[]) are logged but don't stop the run.
# ──────────────────────────────────────────────────────────────────────────────

def validate_writer_inputs(content_row, research_result, config):
    """
    Validates all inputs before the Writer starts.

    Returns: {"valid": bool, "errors": [], "warnings": []}
      errors   → pipeline stops for this row (e.g. no title, bad research)
      warnings → logged but pipeline continues (e.g. missing optional config)
    """
    errors   = []
    warnings = []

    # ── 1. Required content row fields ───────────────────────────────────────
    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")
    if not content_row.get("Keyword"):
        errors.append("Keyword is required in content_row.")
    if not content_row.get("Section"):
        errors.append("Section is required in content_row.")

    # ── 2. Research result validation ────────────────────────────────────────
    # The research_result must be a proper dict from the Researcher Agent.
    # If it's a string, None, or error object, we can't write.
    if not research_result or not isinstance(research_result, dict):
        errors.append(
            f"research_result must be a dict from ResearcherAgent. "
            f"Got: {type(research_result).__name__}"
        )
    else:
        status = research_result.get("status", "")

        # Blocked research = hard error (Gemini API failed in Researcher)
        if "BLOCKED" in status:
            errors.append(
                f"Cannot write: research is BLOCKED. "
                f"Errors: {research_result.get('errors', [])}"
            )
        # Insufficient sources = hard error (not enough quality data to write from)
        elif "insufficient sources" in status:
            errors.append(
                "Cannot write: research has insufficient sources. "
                "Resolve in Researcher before writing."
            )

        # Minimum viable facts — Writer needs at least 3 to write credibly
        facts = research_result.get("key_facts", [])
        if len(facts) < 3:
            errors.append(
                f"Research brief has only {len(facts)} facts (minimum 3 required)."
            )

        # Minimum viable citations — ensures the post can be properly sourced
        citations = research_result.get("citations", [])
        if len(citations) < 3:
            errors.append(
                f"Research brief has only {len(citations)} citations (minimum 3 required)."
            )

    # ── 3. Brand config validation ────────────────────────────────────────────
    # NOTE: This block is intentionally OUTSIDE the research_result else block.
    # In the original code it was accidentally nested inside — that was a bug.
    # Brand config must be validated regardless of research status.
    brand = config.get("brand", {})

    if not brand.get("brand_voice_summary"):
        warnings.append("brand_voice_summary missing — Writer will use default tone.")

    # disclaimer_text replaces fda_disclaimer_text (now industry-agnostic)
    if not brand.get("disclaimer_text"):
        warnings.append(
            "disclaimer_text missing from Config_Brand — "
            "Writer will use a hardcoded FDA default."
        )
    if not brand.get("default_word_count_min"):
        warnings.append("default_word_count_min missing — defaulting to 800.")
    if not brand.get("default_word_count_max"):
        warnings.append("default_word_count_max missing — defaulting to 1200.")
    if not brand.get("content_language"):
        warnings.append("content_language missing — defaulting to English (en).")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ──────────────────────────────────────────────────────────────────────────────
# RESEARCH CONTEXT EXTRACTOR
# Converts the Researcher's structured JSON dict into clean,
# readable text blocks that can be pasted directly into the Gemini prompt.
# ──────────────────────────────────────────────────────────────────────────────

def extract_research_context(research_result):
    """
    Formats the Research Brief dict into prompt-ready text blocks.

    Returns a dict of strings:
      facts_block      : numbered list of facts with confidence + source
      citations_block  : list of citations with title, year, URL
      boundaries_block : list of what NOT to claim
      entities_text    : comma-separated semantic terms for natural writing
      warnings_block   : research caveats for the Writer to keep in mind
    """
    # ── Facts ─────────────────────────────────────────────────────────────────
    facts = research_result.get("key_facts", [])
    facts_block = "\n".join([
        f"- [{f.get('confidence', 'medium').upper()}] {f.get('fact', '')} "
        f"(Source: {f.get('source_url', 'unverified')})"
        for f in facts
    ]) or "No facts provided."

    # ── Citations ─────────────────────────────────────────────────────────────
    citations = research_result.get("citations", [])
    citations_block = "\n".join([
        f"- {c.get('title', 'Untitled')} ({c.get('year', 'unknown')}) "
        f"[{c.get('source_type', 'study')}]: {c.get('url', '')}"
        for c in citations
    ]) or "No citations provided."

    # ── Claim boundaries (what NOT to write) ─────────────────────────────────
    boundaries = research_result.get("claim_boundaries", [])
    boundaries_block = "\n".join([f"- {b}" for b in boundaries]) \
        or "No specific boundaries noted — apply general compliance rules."

    # ── Semantic entities (terminology for natural, credible writing) ─────────
    entities = research_result.get("semantic_entities", [])
    entities_text = ", ".join(entities) if entities else "general wellness terms"

    # ── Research warnings (scientific caveats) ────────────────────────────────
    warnings = research_result.get("warnings", [])
    warnings_block = "\n".join([f"- {w}" for w in warnings]) \
        or "No research warnings."

    return {
        "facts_block":      facts_block,
        "citations_block":  citations_block,
        "boundaries_block": boundaries_block,
        "entities_text":    entities_text,
        "warnings_block":   warnings_block,
    }


# ──────────────────────────────────────────────────────────────────────────────
# COMPLIANCE CHECKER
# Scans the final HTML for disease-claim language that would violate
# the compliance_framework (FDA, FTC, etc.) set in Config_Brand.
#
# SMART: strips the disclaimer block before checking — the disclaimer itself
# contains words like "treat", "cure", "prevent" by design, and should never
# trigger a false HARD FAIL.
#
# Product names are read dynamically from Config_Brand (not hardcoded),
# so this works for any brand in any industry.
# ──────────────────────────────────────────────────────────────────────────────

def check_compliance(html, config=None):
    """
    Checks HTML for disease-claim language violations.

    html   : the raw HTML string from Gemini
    config : full config dict — used to read product names dynamically

    Returns: {"passed": bool, "violations": [str]}
    """
    brand = (config or {}).get("brand", {})

    # ── Step 1: Remove disclaimer before scanning ─────────────────────────────
    # The disclaimer block intentionally contains "treat", "cure", "prevent",
    # "diagnose" — we must strip it before checking the body content.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(class_=["fda-disclaimer", "disclaimer"]):
        tag.decompose()
    for p in soup.find_all("p"):
        if "food and drug administration" in p.get_text().lower():
            p.decompose()

    clean_text = soup.get_text().lower()

    # ── Step 2: Build product names pattern dynamically ───────────────────────
    # Read product names from Config_Brand instead of hardcoding "GloRest" etc.
    # Example: "GloRest, GloSerene, GloBalance" → "GloRest|GloSerene|GloBalance"
    raw_products  = brand.get("product_names",  "")
    raw_brand     = brand.get("brand_name",     "")

    product_list  = [p.strip() for p in raw_products.split(",") if p.strip()]
    if raw_brand and raw_brand not in product_list:
        product_list.append(raw_brand)

    # If no products configured, use a generic catch-all
    product_pattern = "|".join(product_list) if product_list else "this product|our supplement"

    # ── Step 3: Disease claim patterns ───────────────────────────────────────
    # These patterns match language that violates health supplement regulations.
    # Each pattern is specific to avoid false positives (e.g. "treats you well"
    # is fine, "treats insomnia" is not).
    disease_claim_patterns = [
        # treat/cure/heal/reverse + specific disease object
        r"\b(treats?|treating|treated)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(cures?|cured|curing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(heals?|healed|healing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(reverses?|reversed|reversing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",

        # prevent + specific disease (not "prevent fatigue" — only actual diseases)
        r"\b(prevents?|prevented|preventing)\s+(disease|cancer|diabetes|depression|disorder|dementia)\b",

        # diagnose — always a hard block regardless of context
        r"\b(diagnoses|diagnosed|diagnosing)\b",

        # eliminates + disease object within 30 characters
        r"\b(eliminates?|eliminated)\b.{0,30}\b(disease|condition|disorder)\b",

        # Literal strong claim phrases
        r"works like a prescription",
        r"clinically proven to (treat|cure|prevent)",
        r"guaranteed (results|to work)",

        # Product name + disease within 60 characters (dynamic, not hardcoded)
        rf"({product_pattern}).{{0,60}}(insomnia|depression|anxiety disorder|diabetes|cancer|dementia)",
    ]

    violations = []
    for pattern in disease_claim_patterns:
        try:
            matches = re.findall(pattern, clean_text)
            if matches:
                violations.append(
                    f"Potential disease claim detected — pattern: '{pattern[:60]}...'"
                )
        except re.error as e:
            # A bad regex pattern shouldn't crash the whole check
            print(f"   ⚠️  Compliance regex error (skipped): {e}")

    return {"passed": len(violations) == 0, "violations": violations}


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSER
# Cleans the raw HTML from Gemini and validates its structure and word count.
# Word count over max → logs a clear rewrite instruction (not a hard error).
# The Reviewer will pick this up and send a fix back to the Writer if needed.
# ──────────────────────────────────────────────────────────────────────────────

def parse_writer_output(raw_html, config):
    """
    Cleans Gemini's HTML output and checks structure + word count.

    Returns:
      {
        "html":       str   — cleaned, well-formed HTML
        "word_count": int   — body word count (excluding disclaimer)
        "warnings":   list  — structural or length issues found
      }

    Word count rules:
      Under min → warning (Reviewer will request expansion)
      Over max  → warning with explicit fix instruction (not a hard stop)
                  The retry loop in the Orchestrator handles the rewrite.
    """
    # Strip markdown fences if Gemini wrapped the HTML in ```html ... ```
    clean = re.sub(r"^```html\s*", "", raw_html, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$",    "", clean)
    clean = clean.strip()

    # Parse with BeautifulSoup for well-formedness
    soup       = BeautifulSoup(clean, "html.parser")
    clean_html = str(soup)
    word_count = len(soup.get_text().split())

    # ── Structural checks ─────────────────────────────────────────────────────
    warnings = []
    if not soup.find(["h2", "h3"]):
        warnings.append("No H2/H3 headings found — rewrite with proper subheadings.")
    if "sources" not in soup.get_text().lower() \
            and "references" not in soup.get_text().lower():
        warnings.append("No Sources/References section found — add a sources list.")
    if not soup.find("p"):
        warnings.append("No paragraph tags found — HTML structure is malformed.")

    # Check for required HTML sections
    if not soup.find(class_="faq"):
        warnings.append("FAQ section missing — add <section class='faq'>.")
    if not soup.find(class_="hook"):
        warnings.append("Opening hook paragraph missing — add <p class='hook'>.")

    # ── Word count check ──────────────────────────────────────────────────────
    brand    = config.get("brand", {})
    word_min = int(brand.get("default_word_count_min", 800))
    word_max = int(brand.get("default_word_count_max", 1200))

    if word_count < word_min:
        warnings.append(
            f"Word count ({word_count}) is below minimum ({word_min}). "
            f"Expand body sections with more detail — do not add filler."
        )
    elif word_count > word_max:
        # Over the limit → explicit rewrite instruction for the retry loop.
        # This is NOT a hard stop — the Reviewer will send this back as a fix.
        warnings.append(
            f"Word count ({word_count}) exceeds maximum ({word_max}). "
            f"Rewrite and shorten by {word_count - word_max} words. "
            f"Trim body paragraphs — do not cut citations or the FAQ section."
        )

    return {"html": clean_html, "word_count": word_count, "warnings": warnings}


# ══════════════════════════════════════════════════════════════════════════════
# WRITER AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class WriterAgent:

    def __init__(self, config=None):
        """
        Initializes the Writer Agent.
        Reads API keys, model, and brand config from the Orchestrator.
        """
        print("✍️  Initializing Writer Agent...")
        self.config = config or {"brand": {}, "system": {}, "agent_rules": {}}

        # ── Gemini client ─────────────────────────────────────────────────────
        # Model priority: Config_System tab → GEMINI_MODEL env var → default
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("   ❌ GOOGLE_API_KEY missing — Writer will fail.")

        self.client = genai.Client(api_key=api_key)

        self.model_name = (
            self.config.get("system", {}).get("Gemini_Model")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        print(f"   🤖 Model: {self.model_name}")

    # ──────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # Assembles the full Gemini writing prompt from Config_Brand.
    # Nothing is hardcoded — brand, audience, compliance, language, tone,
    # product names, competitor names, and word count all come from the Sheet.
    #
    # On retry: appends a REVISION MODE block with the previous draft
    # and the specific fixes the Reviewer requested.
    # ──────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, content_row, research_ctx, config,
                    previous_draft="", required_fixes=None):
        brand = config.get("brand", {})

        title   = content_row.get("Title",   "")
        keyword = content_row.get("Keyword", "")
        summary = content_row.get("Summary", "")
        section = content_row.get("Section", "")

        word_min = brand.get("default_word_count_min", "800")
        word_max = brand.get("default_word_count_max", "1200")

        brand_name  = brand.get("brand_name",          "the brand")
        brand_voice = brand.get("brand_voice_summary",
                        "Empathetic, direct, and empowering.")
        cta_style   = brand.get("cta_style",           "soft — educational, not pushy")
        tone        = brand.get("tone_formality",       "conversational")

        age_range      = brand.get("audience_age_range",      "38–55")
        gender         = brand.get("audience_gender",         "female")
        pain_points    = brand.get("audience_pain_points",    "fatigue, brain fog, weight gain")
        sophistication = brand.get("audience_sophistication", "educated, reads labels")

        # ── Writer persona from Config_Brand ──────────────────────────────────────────
        # This is the "character" Gemini adopts from the very first token.
        # If writer_persona is defined in the Sheet, use it exactly.
        # If not, build a fallback automatically from the other audience fields.
        writer_persona = brand.get("writer_persona", "")

        # Fallback: construct the persona from existing fields if writer_persona is empty
        if not writer_persona:
            writer_persona = (
                f"A real {gender} in her {age_range.split('–')[0]}s "
                f"who personally deals with {pain_points} "
                f"and now writes honestly about it for others like her. "
                f"Voice: {brand_voice}"
            )

        disclaimer  = brand.get(
            "disclaimer_text",
            "*These statements have not been evaluated by the Food and Drug Administration. "
            "This product is not intended to diagnose, treat, cure, or prevent any disease."
        )
        compliance  = brand.get("compliance_framework", "FDA")
        industry    = brand.get("industry",             "supplements")
        competitors  = brand.get("competitor_names", "")
        avoid_topics = brand.get("avoid_topics",     "competitor names, political topics")

        brand_banned = brand.get("banned_phrases", "delve, leverage, unlock, navigate")
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
            f"{brand_banned}, {compliance_banned}, \"in conclusion\", "
            "\"it's important to note\", \"a holistic approach\", "
            "\"groundbreaking\", \"game-changer\""
        )

        language = brand.get("content_language", "en")
        language_names = {
            "en": "English", "es": "Spanish", "pt": "Portuguese",
            "fr": "French",  "de": "German",  "it": "Italian"
        }
        language_label = language_names.get(language.lower(), language)
        products = brand.get("product_names", "")

        revision_block = ""
        if previous_draft and required_fixes:
            fixes_text = "\n".join([f"- {f}" for f in required_fixes])
            revision_block = f"""

    ════════════════════════════════════════
    REVISION MODE — previous draft was rejected
    ════════════════════════════════════════
    The previous draft failed review. You MUST fix ALL of the following issues:

    {fixes_text}

    PREVIOUS DRAFT (for context only — do NOT copy any part of it):
    {previous_draft[:3000]}

    Write a completely new version that fixes every issue above.
    Keep the same topic, title, keyword, and HTML structure.
    ════════════════════════════════════════"""

        # ── THE PERSONA ANCHOR — must be the very first line Gemini reads ────
        # writer_persona comes from Config_Brand → writer_persona field.
        # If the field is empty, the fallback above already built it from
        # audience_gender + audience_age_range + audience_pain_points.
        # The content of the f-string must be flush to the left margin —
        # any leading spaces here become part of the text Gemini receives.
        return f"""IMPORTANT: You are not writing as an AI.
{writer_persona}
Write exactly that way — imperfect, direct, warm.
Never write like a chatbot or a medical brochure.

You are an expert {industry} copywriter for {brand_name}.

════════════════════════════════════════
LANGUAGE
════════════════════════════════════════
Write ALL content exclusively in {language_label}.
All HTML tags and class names must stay in English (they are code).
All visible text — headings, paragraphs, FAQ, sources — must be in {language_label}.

════════════════════════════════════════
BRAND & AUDIENCE CONTEXT
════════════════════════════════════════
Brand              : {brand_name}
Industry           : {industry}
Products           : {products}
Audience           : {gender}, age {age_range}
Pain points        : {pain_points}
Sophistication     : {sophistication}
Tone               : {tone}
CTA style          : {cta_style}

BRAND VOICE:
{brand_voice}

════════════════════════════════════════
WRITING INPUTS
════════════════════════════════════════
TITLE              : {title}
PRIMARY KEYWORD    : {keyword}
SECTION            : {section}
INTENT / SUMMARY   : {summary}

VERIFIED FACTS (use these — do not invent new claims):
{research_ctx['facts_block']}

CITATIONS (cite these inline in the body):
{research_ctx['citations_block']}

WHAT NOT TO CLAIM (from research team):
{research_ctx['boundaries_block']}

SCIENTIFIC TERMINOLOGY TO USE NATURALLY:
{research_ctx['entities_text']}

RESEARCH WARNINGS (keep these in mind while writing):
{research_ctx['warnings_block']}

════════════════════════════════════════
WRITE LIKE A HUMAN — ANTI-AI-DETECTION RULES
These are the most important rules in this entire prompt.
AI detectors flag content that is too predictable and too uniform.
You must actively break those patterns on every paragraph.
════════════════════════════════════════
SENTENCE LENGTH — BURSTINESS IS MANDATORY:
- Mix extremely short sentences with longer ones in every section.
- Short sentences hit hard. They create rhythm. Use them often.
- Then follow with a longer sentence that expands on the idea and gives
  the reader context, detail, or a nuance they weren't expecting.
- Never write three sentences in a row that are the same length.
- Occasionally use a one-word or two-word sentence. Like this. Seriously.

NATURAL VOICE — SOUND LIKE A REAL PERSON:
- Use contractions everywhere: "you're", "it's", "doesn't", "they're",
  "can't", "won't", "that's", "we've".
- Start sentences with "But", "And", "So", "Yet", "Because".
  Real people do this. AI almost never does. It's a strong human signal.
- Use em-dashes for natural interruptions — like this — mid-sentence.
- Use parentheses for casual asides (the way you'd speak out loud).
- Ask the reader direct rhetorical questions mid-paragraph. Sound familiar?

IMPERFECT LOGIC — HUMANS DON'T OVER-EXPLAIN:
- Don't tie every paragraph into a perfect conclusion.
- Leave some ideas slightly open. Trust the reader to connect the dots.
- Occasionally acknowledge complexity: "It's not a simple answer."
  "The research is mixed on this one." "This varies a lot for everyone."
- Disagree with the obvious sometimes: "The common advice is X — but
  that's not always the full story."

SPECIFIC DETAILS — NOT GENERIC STATEMENTS:
- Use real numbers and timeframes: "In one 2023 study of 412 women..."
  "In as little as 4–6 weeks..." "For most women over 40..."
- Reference relatable scenarios using the audience's own language: {pain_points}
- Concrete and specific — never vague or abstract.

BANNED AI PATTERNS — NEVER USE THESE:
- Never start a paragraph with: "Furthermore", "Moreover", "In addition",
  "It is worth noting", "It is important to understand", "Notably",
  "Interestingly", "Certainly", "Absolutely", "Of course".
- Never end a section with a summary sentence that repeats what was just said.
- Never use perfectly parallel bullet structure (all same length = AI flag).
- Never use: "comprehensive", "crucial", "vital", "robust", "dive into",
  "delve", "leverage", "navigate", "unlock", "holistic", "empower".

TONAL VARIATION — NOT FLAT:
- Start warm, get a little direct in the middle, end encouraging.
- Let mild frustration or surprise come through naturally:
  "And honestly? That's exhausting." / "Here's the thing nobody tells you."
- Vary formality: mostly conversational, but one or two precise clinical
  terms (from the entities list) used naturally show credibility.

STRUCTURE — BREAK THE PATTERN:
- Not every H2 section needs the same number of paragraphs.
- One section can be 1 paragraph. Another can be 3. This is intentional.
- FAQ answers should sound like talking: "Short answer: yes, but it depends."
════════════════════════════════════════

════════════════════════════════════════
ENGAGEMENT RULES (non-negotiable)
════════════════════════════════════════
- Open with a relatable hook — something this woman has experienced.
- Answer the core question directly in the first 40–60 words.
- Short paragraphs: 2 to 3 sentences MAX.
- Subheadings = questions a real woman in this audience would Google.
- Include a "What This Means for You" section with warm, practical advice.
- End with encouragement — not a hard sell.
- Required hedging language for all health claims: {hedge_words}

BANNED WORDS / PHRASES: {all_banned}

════════════════════════════════════════
COMPLIANCE ({compliance})
════════════════════════════════════════
- NEVER make disease claims (treat / cure / prevent / diagnose / reverse).
- ALLOWED: "supports", "may help", "research suggests", "can contribute to".
- NEVER mention competitors: {competitors}
- AVOID these topics entirely: {avoid_topics}
- EVERY objective health claim needs: <span class="citation">[Source, Year]</span>

════════════════════════════════════════
HTML STRUCTURE — all 7 sections are required
════════════════════════════════════════
Output ONLY raw HTML. No markdown. No backticks. No <html> or <body> tags.

1. <p class="hook">Relatable opening hook (1–2 sentences)</p>

2. <p class="answer-first">Direct answer in 40–60 words</p>

3. 2–3 body sections (each with a question-style H2):
   <h2>[Question she would actually Google]</h2>
   <p>2–3 sentence paragraph with <span class="citation">[Source, Year]</span></p>

4. <h2>What This Means for You</h2>
   <p>Warm, practical paragraph. One soft CTA if appropriate.</p>

5. <section class="faq">
     <h2>Frequently Asked Questions</h2>
     <div class="faq-item">
       <h3>[Question]</h3>
       <p>[Answer — 2 to 3 sentences]</p>
     </div>
     <!-- Exactly 3 faq-items total -->
   </section>

6. <section class="sources">
     <h2>Sources</h2>
     <ul>
       <li><a href="[URL]" target="_blank">[Title] ([Year])</a></li>
     </ul>
   </section>

7. <p class="fda-disclaimer">{disclaimer}</p>

════════════════════════════════════════
WORD COUNT: {word_min}–{word_max} words
(body content only — disclaimer and sources do not count)
════════════════════════════════════════{revision_block}"""



    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # Called by the Orchestrator for each row that needs a draft written.
    # On the first attempt: previous_draft and required_fixes are empty.
    # On retry attempts: receives the rejected draft + Reviewer's fix list.
    # ──────────────────────────────────────────────────────────────────────────

    def write_draft(self, content_row, research_result, config=None,
                    previous_draft="", required_fixes=None):
        """
        Full writing pipeline for one Content_Plan row.

        Parameters:
          content_row     : dict from the Content_Plan Sheet row
          research_result : dict output from ResearcherAgent
          config          : full config dict from the Orchestrator
          previous_draft  : rejected HTML from previous attempt (empty = first run)
          required_fixes  : list of fixes from Reviewer (empty = first run)

        Returns a dict:
          {
            "status":     "Draft Complete" | "Needs Review" | "HARD_FAIL" | "BLOCKED"
            "html":       str  — clean HTML draft
            "word_count": int
            "warnings":   list — structural/length issues (non-blocking)
            "violations": list — compliance violations (only if HARD_FAIL)
            "errors":     list — hard failures (only if BLOCKED)
          }
        """
        cfg   = config if config is not None else self.config
        title = content_row.get("Title", "untitled")
        print(f"\n✍️  Writer running for: {title}")

        # ── Step 1: Validate all inputs before touching Gemini ────────────────
        validation = validate_writer_inputs(content_row, research_result, cfg)
        if not validation["valid"]:
            print("   🚨 Writer BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"      ❌ {err}")
            return {
                "status":     "BLOCKED",
                "errors":     validation["errors"],
                "warnings":   validation["warnings"],
                "html":       "",
                "word_count": 0,
                "violations": []
            }

        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"   ⚠️  {w}")

        # ── Step 2: Extract research context into prompt-ready text blocks ────
        research_ctx = extract_research_context(research_result)

        # ── Step 3: Build the Gemini prompt ───────────────────────────────────
        prompt = self._build_prompt(
            content_row    = content_row,
            research_ctx   = research_ctx,
            config         = cfg,
            previous_draft = previous_draft,
            required_fixes = required_fixes or []
        )

        # ── Step 4: Call Gemini ───────────────────────────────────────────────
        print(f"   🤖 Calling Gemini ({self.model_name})...")
        try:
            raw_html = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt
            ).text.strip()
        except Exception as e:
            print(f"   🚨 Gemini API error: {e}")
            return {
                "status":     "BLOCKED",
                "errors":     [f"Gemini API error: {str(e)}"],
                "warnings":   [],
                "html":       "",
                "word_count": 0,
                "violations": []
            }

        # ── Step 5: Compliance check ──────────────────────────────────────────
        # Scans the body content for disease-claim language.
        # Strips the disclaimer block first to prevent false positives.
        # If violations are found → HARD_FAIL (human must review before retrying)
        compliance = check_compliance(raw_html, config=cfg)
        if not compliance["passed"]:
            print("   🚨 HARD FAIL: Disease claim language detected.")
            for v in compliance["violations"]:
                print(f"      ❌ {v}")
            return {
                "status":     "HARD_FAIL",
                "errors":     [],
                "warnings":   [],
                "html":       raw_html,   # preserve for human inspection
                "word_count": 0,
                "violations": compliance["violations"]
            }

        # ── Step 6: Parse and validate output structure + word count ──────────
        parsed       = parse_writer_output(raw_html, cfg)
        all_warnings = validation["warnings"] + parsed["warnings"]

        # ── Step 7: Determine final status ───────────────────────────────────
        # Any warnings (structural issues, word count) → "Needs Review"
        # The Orchestrator's retry loop will send these to the Reviewer,
        # and if the Reviewer flags them, the Writer will rewrite.
        # No warnings → "Draft Complete" — moves to Reviewer for final approval.
        status = "Draft Complete" if not all_warnings else "Needs Review"

        print(f"   ✅ Writing complete.")
        print(f"      Status     : {status}")
        print(f"      Word count : {parsed['word_count']}")
        if all_warnings:
            for w in all_warnings:
                print(f"      ⚠️  {w}")

        return {
            "status":     status,
            "html":       parsed["html"],
            "word_count": parsed["word_count"],
            "warnings":   all_warnings,
            "violations": [],
            "errors":     []
        }
