import os
import re
import json
from google import genai
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────

def validate_writer_inputs(content_row, research_result, config):
    """
    Hard-gate validation before any writing begins.
    Returns {"valid": bool, "errors": [], "warnings": []}
    """
    errors   = []
    warnings = []

    # 1. Content row fields
    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")
    if not content_row.get("Keyword"):
        errors.append("Keyword is required in content_row.")
    if not content_row.get("Section"):
        errors.append("Section is required in content_row.")

    # 2. Research result must be a valid dict (not a string, None, or error)
    if not research_result or not isinstance(research_result, dict):
        errors.append(
            "research_result must be a dict from ResearcherAgent. "
            f"Got: {type(research_result).__name__}"
        )
    else:
        # Research must have passed (not BLOCKED)
        status = research_result.get("status", "")
        if "BLOCKED" in status:
            errors.append(
                f"Cannot write: research is BLOCKED. "
                f"Errors: {research_result.get('errors', [])}"
            )
        elif "insufficient sources" in status:
            errors.append(
                "Cannot write: research has insufficient sources. "
                "Resolve in Researcher before writing."
            )

        # Minimum viable facts
        facts = research_result.get("key_facts", [])
        if len(facts) < 3:
            errors.append(
                f"Research brief has only {len(facts)} facts (minimum 3 required)."
            )

        # Minimum viable citations
        citations = research_result.get("citations", [])
        if len(citations) < 3:
            errors.append(
                f"Research brief has only {len(citations)} citations (minimum 3 required)."
            )

    # 3. Brand config
        brand = config.get("brand", {})
        if not brand.get("brand_voice_summary"):
            warnings.append("brand_voice_summary missing — Writer will use default tone.")
        if not brand.get("fda_disclaimer_text"):
            warnings.append(
                "fda_disclaimer_text missing from Config_Brand — "
                "Writer will use hardcoded default."
            )
        if not brand.get("default_word_count_min"):
            warnings.append("default_word_count_min missing — defaulting to 300.")
        if not brand.get("default_word_count_max"):
            warnings.append("default_word_count_max missing — defaulting to 500.")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ─────────────────────────────────────────────
# RESEARCH CONTEXT EXTRACTOR
# ─────────────────────────────────────────────

def extract_research_context(research_result):
    """
    Converts the Researcher's structured dict into
    clean, prompt-ready text blocks for the Writer.
    """
    # Facts block
    facts = research_result.get("key_facts", [])
    facts_block = "\n".join([
        f"- [{f.get('confidence', 'medium').upper()}] {f.get('fact', '')} "
        f"(Source: {f.get('source_url', 'unverified')})"
        for f in facts
    ]) or "No facts provided."

    # Citations block
    citations = research_result.get("citations", [])
    citations_block = "\n".join([
        f"- {c.get('title', 'Untitled')} ({c.get('year', 'unknown')}) "
        f"[{c.get('source_type', 'unknown')}]: {c.get('url', '')}"
        for c in citations
    ]) or "No citations provided."

    # Claim boundaries
    boundaries = research_result.get("claim_boundaries", [])
    boundaries_block = "\n".join([f"- {b}" for b in boundaries]) \
        or "No specific boundaries noted — apply general compliance rules."

    # Semantic entities (AEO terminology)
    entities = research_result.get("semantic_entities", [])
    entities_text = ", ".join(entities) if entities else "general wellness terms"

    # Warnings from researcher
    warnings = research_result.get("warnings", [])
    warnings_block = "\n".join([f"- {w}" for w in warnings]) \
        or "No research warnings."

    return {
        "facts_block":      facts_block,
        "citations_block":  citations_block,
        "boundaries_block": boundaries_block,
        "entities_text":    entities_text,
        "warnings_block":   warnings_block,
        "raw_brief":        research_result.get("raw_brief", "")
    }


# ─────────────────────────────────────────────
# COMPLIANCE CHECKER
# ─────────────────────────────────────────────

def check_compliance(html):
    """
    Checks HTML for disease-claim language.
    Smart: strips the FDA disclaimer block before checking
    so the disclaimer itself doesn't trigger a false HARD FAIL.
    Returns {"passed": bool, "violations": [str]}
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove disclaimer paragraph before checking
    # (it contains "treat", "cure", "prevent", "diagnose" by design)
    for tag in soup.find_all(class_=["fda-disclaimer", "disclaimer"]):
        tag.decompose()

    # Also remove any <p> that contains the exact FDA disclaimer text
    for p in soup.find_all("p"):
        if "food and drug administration" in p.get_text().lower():
            p.decompose()

    clean_text = soup.get_text().lower()

    disease_claim_patterns = [
        # Treat/cure/heal — only blocked if followed by a disease/condition object
        r"\b(treats?|treating|treated)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(cures?|cured|curing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(heals?|healed|healing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(reverses?|reversed|reversing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",

        # Prevent — only blocked with specific disease target
        r"\b(prevents?|prevented|preventing)\s+(disease|cancer|diabetes|depression|disorder|dementia)\b",

        # Diagnose — always a hard block regardless of context
        r"\b(diagnoses|diagnosed|diagnosing)\b",

        # Eliminates + disease object within 30 characters
        r"\b(eliminates?|eliminated)\b.{0,30}\b(disease|condition|disorder)\b",

        # Explicit strong claim phrases — literal match
        r"works like a prescription",
        r"clinically proven to (treat|cure|prevent)",
        r"guaranteed (results|to work)",

        # Brand name + disease within 60 characters — always a claim
        r"(this product|our supplement|GloRest|GloSerene|GloBalance|Glomend).{0,60}(insomnia|depression|anxiety disorder|diabetes|cancer|dementia)",
    ]

    violations = []
    for pattern in disease_claim_patterns:
        matches = re.findall(pattern, clean_text)
        if matches:
            violations.append(
                f"Potential disease claim: '{pattern}' matched in body content."
            )

    return {"passed": len(violations) == 0, "violations": violations}


# ─────────────────────────────────────────────
# OUTPUT PARSER
# ─────────────────────────────────────────────

def parse_writer_output(raw_html, config):
    """
    Cleans and validates the HTML output from Gemini.
    Returns cleaned HTML and word count.
    """
    # Strip markdown fences if Gemini added them
    clean = re.sub(r"^```html\s*", "", raw_html, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$",    "", clean)
    clean = clean.strip()

    # Parse with BeautifulSoup for well-formedness
    soup       = BeautifulSoup(clean, "html.parser")
    clean_html = str(soup)
    word_count = len(soup.get_text().split())

    warnings = []
    if not soup.find(["h2", "h3"]):
        warnings.append("NEEDS REVIEW: No H2/H3 headings found in draft.")
    if "sources" not in soup.get_text().lower() \
            and "references" not in soup.get_text().lower():
        warnings.append("NEEDS REVIEW: No Sources/References section found.")
    if not soup.find("p"):
        warnings.append("NEEDS REVIEW: No paragraph tags found in draft.")

    brand    = config.get("brand", {})
    word_min = int(brand.get("default_word_count_min", 800))
    word_max = int(brand.get("default_word_count_max", 1200))

    if word_count < word_min:
        warnings.append(
            f"NEEDS REVIEW: Word count ({word_count}) is below minimum ({word_min})."
        )
    elif word_count > word_max:
        warnings.append(
            f"NEEDS REVIEW: Word count ({word_count}) exceeds maximum ({word_max})."
        )

    return {"html": clean_html, "word_count": word_count, "warnings": warnings}


# ─────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────

class WriterAgent:
    def __init__(self, config=None):
        """
        Agent 3: Health & Wellness Copywriter.

        Improvements:
        - Accepts structured research dict from ResearcherAgent
        - Uses claim_boundaries from Researcher directly in prompt
        - Engagement-first writing instructions (conversational, empathetic, readable)
        - Smart compliance check (doesn't fail on FDA disclaimer)
        - Structured dict output with status, html, word_count, warnings
        - Full input validation
        - Config-driven word count, disclaimer, brand voice
        """
        print("✍️ Initializing Writer Agent...")
        self.config = config or {"brand": {}}

        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


    # ── PROMPT BUILDER ───────────────────────────────────────────────────
    def _build_prompt(self, content_row, research_ctx, config,previous_draft="", required_fixes=None):
        brand          = config.get("brand", {})
        title          = content_row.get("Title",   "")
        keyword        = content_row.get("Keyword", "")
        summary        = content_row.get("Summary", "")
        section        = content_row.get("Section", "")
        word_min       = brand.get("default_word_count_min", "800")
        word_max       = brand.get("default_word_count_max", "1200")
        brand_voice    = brand.get("brand_voice_summary",
                            "Empathetic, direct, and empowering. "
                            "Peer-to-peer tone — like a knowledgeable friend, "
                            "not a clinical textbook.")
        banned         = brand.get("banned_phrases", "delve, leverage, unlock, navigate")
        fda_disclaimer = brand.get(
                "fda_disclaimer_text",
                "*These statements have not been evaluated by the Food and Drug Administration. "
                "This product is not intended to diagnose, treat, cure, or prevent any disease."
            )

        # ── Build revision block BEFORE the return ──────────────────
        revision_block = ""
        if previous_draft and required_fixes:
            fixes_text = "\n".join([f"- {f}" for f in required_fixes])
            revision_block = f"""     
    

════════════════════════════════════════
REVISION MODE — previous draft failed review
════════════════════════════════════════
The previous draft was rejected. You MUST fix ALL of the following:

{fixes_text}

PREVIOUS DRAFT (rewrite completely — do not copy):
{previous_draft[:3000]}

Write a brand new version that fixes every issue above.
Keep the same topic, title, and HTML structure requirements.
════════════════════════════════════════"""

        # ── Single return with revision_block at the end ─────────────
        return f"""You are an expert health and wellness copywriter for Glomend,
a supplement brand for women navigating perimenopause and menopause (ages 38–55).

You write content that feels like it was written by a trusted, knowledgeable friend —
not a doctor, not a textbook. Women read your posts and think "finally, someone gets it."

════════════════════════════════════════
WRITING INPUTS
════════════════════════════════════════
TITLE          : {title}
PRIMARY KEYWORD: {keyword}
SECTION        : {section}
INTENT/SUMMARY : {summary}

BRAND VOICE:
{brand_voice}

VERIFIED FACTS (use these — do not invent new claims):
{research_ctx['facts_block']}

CITATIONS (cite these inline):
{research_ctx['citations_block']}

WHAT NOT TO CLAIM (from research team):
{research_ctx['boundaries_block']}

SCIENTIFIC TERMINOLOGY TO USE NATURALLY:
{research_ctx['entities_text']}

RESEARCH WARNINGS:
{research_ctx['warnings_block']}

════════════════════════════════════════
ENGAGEMENT RULES (non-negotiable)
════════════════════════════════════════
- Open with a relatable hook.
- Answer the core question in the first 40–60 words.
- Short paragraphs — 2 to 3 sentences MAX.
- Subheadings = questions a real woman would Google.
- Include "What This Means for You" section.
- End with encouragement.

BANNED: {banned}, "in conclusion", "it's important to note",
"a holistic approach", "groundbreaking", "game-changer"

════════════════════════════════════════
COMPLIANCE (hard rules)
════════════════════════════════════════
- NO disease claims (treat/cure/prevent/diagnose).
- ALLOWED: "supports", "may help", "research suggests".
- NO competitors: Goli, Bonafide, Ritual, Menofit, Equelle.
- MANDATORY: every objective claim needs <span class="citation">[Source, Year]</span>.

════════════════════════════════════════
HTML STRUCTURE — all 7 sections required
════════════════════════════════════════
Output ONLY raw HTML. No markdown. No backticks. No <html>/<body> tags.

1. <p class="hook">Relatable opening hook</p>
2. <p class="answer-first">Direct answer in 40–60 words</p>
3. 2–3 body sections:
   <h2>[Question she'd Google]</h2>
   <p>2–3 sentence paragraph with <span class="citation">[Source, Year]</span></p>
4. <h2>What This Means for You</h2><p>Warm practical paragraph</p>
5. <section class="faq">
     <h2>Frequently Asked Questions</h2>
     <div class="faq-item"><h3>[Question]</h3><p>[Answer 2–3 sentences]</p></div>
     (3 faq-items total)
   </section>
6. <section class="sources">
     <h2>Sources</h2>
     <ul><li><a href="[URL]">[Title] ([Year])</a></li></ul>
   </section>
7. <p class="fda-disclaimer">{fda_disclaimer}</p>

════════════════════════════════════════
WORD COUNT: {word_min}–{word_max} words (body content only, not disclaimer)
════════════════════════════════════════{revision_block}"""



    # ── MAIN: write_draft ───────────────────────────────────────────────
    def write_draft(self, content_row, research_result, config=None,
                    previous_draft="", required_fixes=None):
        """
        Full writing pipeline for one Content_Plan row.

        Args:
            content_row     : dict — Title, Keyword, Section, Summary
            research_result : dict — structured output from ResearcherAgent
            config          : dict — from Google Sheets config tabs
                              (falls back to self.config)

        Returns:
            dict with keys:
                status       : "Draft Complete" | "Needs Review" |
                               "HARD_FAIL" | "BLOCKED"
                html         : str — clean HTML draft
                word_count   : int
                warnings     : [str]
                violations   : [str] (only if HARD_FAIL)
                errors       : [str] (only if BLOCKED)
        """
        if config is None:
            config = self.config

        title = content_row.get("Title", "untitled")
        print(f"\n✍️ Writer running for: {title}")

        # ── Step 1: Validate inputs ──────────
        validation = validate_writer_inputs(content_row, research_result, config)
        if not validation["valid"]:
            print("🚨 Writer BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"   ❌ {err}")
            return {
                "status":     "BLOCKED",
                "errors":     validation["errors"],
                "warnings":   validation["warnings"],
                "html":       "",
                "word_count": 0,
                "violations": []
            }

        if validation["warnings"]:
            print("⚠️ Writer warnings:")
            for w in validation["warnings"]:
                print(f"   ⚠ {w}")

        # ── Step 2: Extract research context ──
        research_ctx = extract_research_context(research_result)

        # ── Step 3: Build prompt and call Gemini ──
        prompt = self._build_prompt(
            content_row, research_ctx, config,
            previous_draft=previous_draft,
            required_fixes=required_fixes or []
        )
        print(f"🤖 Calling Gemini ({os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')})...")

        try:
            raw_html = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            ).text.strip()
        except Exception as e:
            print(f"🚨 Gemini API error: {e}")
            return {
                "status":     "BLOCKED",
                "errors":     [f"Gemini API error: {str(e)}"],
                "warnings":   [],
                "html":       "",
                "word_count": 0,
                "violations": []
            }

        # ── Step 4: Compliance check ──────────
        # Smart check: strips FDA disclaimer before scanning
        compliance = check_compliance(raw_html)
        if not compliance["passed"]:
            print("🚨 HARD FAIL: Disease claim language detected!")
            for v in compliance["violations"]:
                print(f"   ❌ {v}")
            return {
                "status":     "HARD_FAIL",
                "errors":     [],
                "warnings":   [],
                "html":       raw_html,   # preserve for human review
                "word_count": 0,
                "violations": compliance["violations"]
            }

        # ── Step 5: Parse and validate output ──
        parsed = parse_writer_output(raw_html, config)

        # Merge warnings
        all_warnings = validation["warnings"] + parsed["warnings"]

        # ── Step 6: Determine final status ────
        status = "Draft Complete"
        if all_warnings:
            status = "Needs Review"

        # ── Step 7: Report ───────────────────
        print(f"✅ Writing complete.")
        print(f"   Status     : {status}")
        print(f"   Word count : {parsed['word_count']}")
        if all_warnings:
            for w in all_warnings:
                print(f"   ⚠ {w}")

        return {
            "status":     status,
            "html":       parsed["html"],
            "word_count": parsed["word_count"],
            "warnings":   all_warnings,
            "violations": [],
            "errors":     []
        }
    def parse_writer_output(raw_html, config):
        # Strip markdown fences
        clean = re.sub(r"^```html\s*", "", raw_html, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$",    "", clean)
        clean = clean.strip()

        soup       = BeautifulSoup(clean, "html.parser")
        clean_html = str(soup)
        word_count = len(soup.get_text().split())

        warnings = []
        if not soup.find(["h2", "h3"]):
            warnings.append("NEEDS REVIEW: No H2/H3 headings found in draft.")
        if "sources" not in soup.get_text().lower() \
                and "references" not in soup.get_text().lower():
            warnings.append("NEEDS REVIEW: No Sources/References section found.")
        if not soup.find("p"):
            warnings.append("NEEDS REVIEW: No paragraph tags found in draft.")

        # ✅ Fixed: use config, not undefined variable
        brand    = config.get("brand", {})
        word_min = int(brand.get("default_word_count_min", 800))
        word_max = int(brand.get("default_word_count_max", 1200))

        if word_count < word_min:
            warnings.append(
                f"NEEDS REVIEW: Word count ({word_count}) is below minimum ({word_min})."
            )
        elif word_count > word_max:
            warnings.append(
                f"NEEDS REVIEW: Word count ({word_count}) exceeds maximum ({word_max})."
            )

        return {"html": clean_html, "word_count": word_count, "warnings": warnings}
