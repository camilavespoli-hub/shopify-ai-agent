import os
import re
import json
from google import genai


# ─────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────

def validate_reviewer_inputs(content_row, draft_text, research_result, config):
    """
    Hard-gate validation before any review begins.
    Returns {"valid": bool, "errors": [], "warnings": []}
    """
    errors   = []
    warnings = []

    # 1. Draft must exist and be substantive
    if not draft_text or not isinstance(draft_text, str):
        errors.append("draft_text is required and must be a string.")
    elif len(draft_text.strip()) < 100:
        errors.append(
            f"Draft is too short ({len(draft_text.strip())} chars). "
            "Minimum 100 characters required."
        )

    # 2. Writer must not have been BLOCKED or HARD_FAIL
    if isinstance(draft_text, str) and draft_text.startswith(
        ("BLOCKED", "HARD_FAIL", "Research failed")
    ):
        errors.append(
            f"Cannot review: Writer returned an error state: "
            f"'{draft_text[:60]}'"
        )

    # 3. Content row fields
    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")

    # 4. Research result for claim_boundaries check
    if not research_result or not isinstance(research_result, dict):
        warnings.append(
            "research_result not provided — "
            "topic-specific claim boundary check will be skipped."
        )

    # 5. Brand config
    brand = config.get("brand", {})
    if not brand.get("brand_voice_summary"):
        warnings.append(
            "brand_voice_summary missing — "
            "AI brand tone check will use defaults."
        )
    if not brand.get("fda_disclaimer_text"):
        warnings.append(
            "fda_disclaimer_text missing from Config_Brand — "
            "Reviewer will use hardcoded default pattern."
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ─────────────────────────────────────────────
# TOOL 1: SMART REGEX COMPLIANCE SCAN
# ─────────────────────────────────────────────

def scan_compliance(text):
    """
    Regex-based disease claim scanner.
    Smart: strips the FDA disclaimer lines before scanning
    so the disclaimer never triggers a false HARD FAIL.

    Returns:
        {
            "has_violations": bool,
            "violations": [str],          # hard fails
            "flags": [str],               # soft flags for AI review
            "clean_text": str             # text with disclaimer removed
        }
    """
    # ── Strip HTML tags before scanning ──────
    text = re.sub(r"<[^>]+>", " ", text)
    
    # ── Strip disclaimer before scanning ─────
    lines = text.split("\n")
    disclaimer_keywords = [
        "food and drug administration",
        "not intended to diagnose",
        "these statements have not been evaluated"
    ]
    clean_lines = [
        line for line in lines
        if not any(kw in line.lower() for kw in disclaimer_keywords)
    ]
    clean_text = "\n".join(clean_lines).lower()

    # ── Hard-fail patterns (definite disease claims) ──
    hard_patterns = [
        (r"\b(treats|treated|treating)\s+\w+",
         "Active disease treatment claim"),
        (r"\b(cures|cured|curing)\s+\w+",
         "Disease cure claim"),
        (r"\b(prevents|prevented|preventing)\s+\w+",
         "Disease prevention claim"),
        (r"\b(diagnoses|diagnosed|diagnosing)\b",
         "Diagnosis claim"),
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
    ]

    # ── Soft-flag patterns (require AI context review) ──
    soft_patterns = [
        (r"\binsomnia\b",
         "Named sleep disorder — verify no disease claim context"),
        (r"\bdepression\b",
         "Named mental health condition — verify context"),
        (r"\banxiety disorder\b",
         "Named disorder — verify no disease claim context"),
        (r"\bmedication\b",
         "Medication reference — verify no drug comparison"),
        (r"\b100%\b|\bguaranteed\b",
         "Absolute claim language — verify context"),
        (r"\bclinically proven\b",
         "Clinical claim — verify substantiation exists"),
    ]

    violations = []
    flags      = []

    for pattern, label in hard_patterns:
        if re.search(pattern, clean_text):
            violations.append(f"HARD FAIL — {label}: pattern '{pattern}'")

    for pattern, label in soft_patterns:
        if re.search(pattern, clean_text):
            flags.append(f"SOFT FLAG — {label}: pattern '{pattern}'")

    return {
        "has_violations": len(violations) > 0,
        "violations":     violations,
        "flags":          flags,
        "clean_text":     clean_text
    }


# ─────────────────────────────────────────────
# TOOL 2: STRUCTURAL CHECKS
# ─────────────────────────────────────────────

def check_structure(text, config):
    """
    Validates the HTML draft for required structural elements.
    Handles both HTML (from Writer) and Markdown gracefully.
    """
    issues = []
    notes  = []
    tl     = text.lower()

    # ── Headings: accept HTML <h2>/<h3> OR Markdown ##/**bold** ──
    has_html_headings = bool(re.search(r"<h[23][^>]*>", tl))
    has_md_headings   = "##" in text or "**" in text
    if not has_html_headings and not has_md_headings:
        issues.append("STRUCTURE: No headings found (## or **bold** or <h2>/<h3>).")

    # ── FAQ: accept <section class="faq">, <h2>...faq, or plain text ──
    has_faq = (
        'class="faq"' in tl
        or "frequently asked questions" in tl
        or "<h2>faq" in tl
        or "faq" in tl
    )
    if not has_faq:
        issues.append("STRUCTURE: FAQ section missing.")

    # ── Sources: accept <section class="sources">, <h2>sources, or plain ──
    has_sources = (
        'class="sources"' in tl
        or "<h2>sources" in tl
        or "sources" in tl
        or "references" in tl
    )
    if not has_sources:
        issues.append("STRUCTURE: Sources/References section missing.")

    # ── FDA Disclaimer: accept <p class="fda-disclaimer"> or plain text ──
    has_fda = (
        'class="fda-disclaimer"' in tl
        or "food and drug administration" in tl
        or "not intended to diagnose" in tl
        or "these statements have not been evaluated" in tl
    )
    if not has_fda:
        issues.append("STRUCTURE: FDA disclaimer is missing.")

    # ── Citations: accept HTML <span class="citation"> OR [Source, Year] ──
    html_citations = len(re.findall(r'class="citation"', tl))
    md_citations   = len(re.findall(r"\[.+?,\s*\d{4}\]", text))
    citation_count = html_citations + md_citations

    if citation_count == 0:
        issues.append(
            "CITATIONS: No inline citations found. "
            "Every objective claim requires a citation."
        )
    elif citation_count < 3:
        notes.append(
            f"CITATIONS: Only {citation_count} inline citation(s) found. "
            "Consider adding more."
        )

    # ── Word count ──
    # Strip HTML tags for accurate word count
    plain_text = re.sub(r"<[^>]+>", " ", text)
    word_count = len(plain_text.split())
    word_min   = int(config.get("brand", {}).get("default_word_count_min", 800))
    word_max   = int(config.get("brand", {}).get("default_word_count_max", 1500))

    if word_count < word_min:
        issues.append(
            f"WORD COUNT: {word_count} words is below minimum ({word_min})."
        )
    elif word_count > word_max:
        notes.append(
            f"WORD COUNT: {word_count} words exceeds maximum ({word_max}). "
            "Consider trimming."
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "notes":  notes
    }



# ─────────────────────────────────────────────
# AI REVIEWER PROMPT BUILDER
# ─────────────────────────────────────────────

def build_reviewer_prompt(draft_text, content_row, research_result,
                           compliance_flags, config):
    """
    Builds the AI review prompt.
    Only called if regex scan passes — AI handles nuance and tone.
    """
    brand        = config.get("brand", {})
    title        = content_row.get("Title",   "")
    section      = content_row.get("Section", "")
    brand_voice  = brand.get(
        "brand_voice_summary",
        "Empathetic, direct, and empowering. "
        "Peer-to-peer tone — like a knowledgeable friend, not a textbook."
    )
    banned       = brand.get(
        "banned_phrases",
        "delve, leverage, unlock, navigate, in conclusion, groundbreaking"
    )

    # Claim boundaries from Researcher
    boundaries = []
    if isinstance(research_result, dict):
        boundaries = research_result.get("claim_boundaries", [])
    boundaries_block = "\n".join([f"- {b}" for b in boundaries]) \
        or "No specific boundaries — apply general FTC/FDA rules."

    # Soft flags from regex to give AI context
    flags_block = "\n".join(compliance_flags) \
        or "No soft flags detected by regex scan."

    return f"""You are a Senior Compliance Officer and Brand Editor for Glomend,
a women's health supplement brand focused on perimenopause and menopause.

You have two jobs: (1) catch any compliance issues the regex scanner missed,
and (2) evaluate brand voice and readability quality.

════════════════════════════════════════
REVIEW CONTEXT
════════════════════════════════════════
ARTICLE TITLE : {title}
SECTION       : {section}
BRAND VOICE   : {brand_voice}

TOPIC-SPECIFIC CLAIM BOUNDARIES (from Research team):
{boundaries_block}

SOFT FLAGS FROM AUTOMATED SCAN (review these in context):
{flags_block}

BANNED PHRASES:
{banned}, "it's important to note", "a holistic approach",
"revolutionary", "game-changer", "take charge", "empower yourself"

DRAFT CONTENT:
{draft_text}

════════════════════════════════════════
REVIEW CHECKLIST — evaluate each item
════════════════════════════════════════

COMPLIANCE (hard rules):
[ ] No disease claims (treat/cure/prevent/diagnose in disease context)
[ ] No competitor brand mentions
[ ] No personalized medical advice or dosage guidance
[ ] No absolute claims without substantiation ("guaranteed", "always works")
[ ] Uncertainty stated honestly where evidence is limited
[ ] Every objective claim has an inline citation [Source, Year]
[ ] Topic-specific claim boundaries above are respected

BRAND VOICE (soft rules):
[ ] Empathetic opening — acknowledges the reader's experience
[ ] Conversational tone — "you", contractions, no clinical distance
[ ] No banned words or phrases
[ ] Short paragraphs (2–3 sentences max)
[ ] Subheadings are questions a real woman would Google
[ ] "What This Means for You" section feels warm, not preachy
[ ] FAQ questions feel authentic, not generic

════════════════════════════════════════
OUTPUT — return ONLY a ```json block
════════════════════════════════════════
{{
  "status": "PASS" | "PASS_WITH_NOTES" | "FAIL",
  "compliance_result": "PASS" | "FAIL",
  "brand_voice_result": "PASS" | "PASS_WITH_NOTES" | "FAIL",
  "violations": [
    "Exact quote or description of compliance violation"
  ],
  "brand_notes": [
    "Specific brand voice or readability suggestion"
  ],
  "required_fixes": [
    "Specific change required before approval (compliance only)"
  ],
  "suggested_improvements": [
    "Optional improvements for tone or engagement"
  ],
  "reviewer_summary": "One sentence summary for the Google Sheet ReviewNotes column."
}}

STATUS RULES:
- PASS          = no violations, no required fixes
- PASS_WITH_NOTES = no violations, but brand notes or suggestions exist
- FAIL          = any compliance violation OR any required_fix exists

If violations or required_fixes are empty, use empty arrays [].
Do NOT output anything outside the ```json block."""


# ─────────────────────────────────────────────
# OUTPUT PARSER
# ─────────────────────────────────────────────

def parse_reviewer_output(raw_text):
    """
    Parses the AI reviewer's JSON output.
    Falls back gracefully if JSON is malformed.
    """
    json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            pass  # Fall through to fallback

    # Fallback: treat entire response as a note
    return {
        "status":                  "PASS_WITH_NOTES",
        "compliance_result":       "PASS",
        "brand_voice_result":      "PASS_WITH_NOTES",
        "violations":              [],
        "brand_notes":             ["NEEDS REVIEW: AI reviewer returned unstructured output."],
        "required_fixes":          [],
        "suggested_improvements":  [],
        "reviewer_summary":        f"AI review returned unstructured output: {raw_text[:100]}"
    }


# ─────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────

class ReviewerAgent:
    def __init__(self, config=None):
        """
        Agent 4: Compliance & Brand Guardrail.

        Two-pass review:
        Pass 1 — Automated: regex compliance scan + structural checks
        Pass 2 — AI: nuanced compliance + brand voice + engagement quality

        Output: structured dict with status field for Orchestrator routing.
        Accepts plain Markdown text from Writer (not HTML).
        """
        print("🛡️ Initializing Reviewer Agent...")
        self.config = config or {"brand": {}}

        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


        # FDA disclaimer pattern for presence check
        self.fda_pattern = re.compile(
            r"statements.*not.*evaluated.*food.*and.*drug.*administration"
            r".*not.*intended.*diagnose.*treat.*cure.*prevent.*disease",
            re.IGNORECASE | re.DOTALL
        )

    # ── MAIN: review_draft ──────────────────────────────────────────────
    def review_draft(self, content_row, draft_text, research_result, config=None):
        """
        Full two-pass review pipeline.

        Args:
            content_row     : dict — Title, Keyword, Section from Content_Plan
            draft_text      : str  — plain Markdown from WriterAgent
            research_result : dict — structured output from ResearcherAgent
            config          : dict — from Google Sheets config tabs

        Returns:
            dict with keys:
                status                : "PASS" | "PASS_WITH_NOTES" |
                                        "FAIL" | "HARD_FAIL" | "BLOCKED"
                compliance_result     : "PASS" | "FAIL"
                brand_voice_result    : "PASS" | "PASS_WITH_NOTES" | "FAIL"
                violations            : [str] — hard compliance failures
                brand_notes           : [str] — tone/engagement suggestions
                required_fixes        : [str] — must fix before approval
                suggested_improvements: [str] — optional improvements
                reviewer_summary      : str   — one line for Sheet ReviewNotes
                structural_issues     : [str] — from structural check
                warnings              : [str] — non-blocking notes
                errors                : [str] — only if BLOCKED
        """
        if config is None:
            config = self.config

        title = content_row.get("Title", "untitled")
        print(f"\n🛡️ Reviewer running for: {title}")

        # ── Step 1: Validate inputs ──────────
        validation = validate_reviewer_inputs(
            content_row, draft_text, research_result, config
        )
        if not validation["valid"]:
            print("🚨 Reviewer BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"   ❌ {err}")
            return {
                "status":                 "BLOCKED",
                "compliance_result":      "UNKNOWN",
                "brand_voice_result":     "UNKNOWN",
                "violations":             [],
                "brand_notes":            [],
                "required_fixes":         [],
                "suggested_improvements": [],
                "reviewer_summary":       "; ".join(validation["errors"]),
                "structural_issues":      [],
                "warnings":               validation["warnings"],
                "errors":                 validation["errors"]
            }

        # ── Step 2: Automated regex scan ─────
        print("   🔍 Pass 1: Automated compliance scan...")
        compliance_scan = scan_compliance(draft_text)

        # Hard fail — don't proceed to AI, flag immediately
        if compliance_scan["has_violations"]:
            print("🚨 HARD FAIL: Disease claim language detected!")
            for v in compliance_scan["violations"]:
                print(f"   ❌ {v}")
            return {
                "status":                 "HARD_FAIL",
                "compliance_result":      "FAIL",
                "brand_voice_result":     "UNKNOWN",
                "violations":             compliance_scan["violations"],
                "brand_notes":            [],
                "required_fixes":         compliance_scan["violations"],
                "suggested_improvements": [],
                "reviewer_summary":       (
                    f"HARD FAIL: {len(compliance_scan['violations'])} "
                    "disease claim violation(s) detected. Requires Writer revision."
                ),
                "structural_issues":      [],
                "warnings":               compliance_scan["flags"],
                "errors":                 []
            }

        # ── Step 3: Structural checks ─────────
        print("   📋 Pass 1: Structural validation...")
        structure = check_structure(draft_text, config)
        if not structure["passed"]:
            print("   ⚠️ Structural issues found:")
            for issue in structure["issues"]:
                print(f"      - {issue}")

        # ── Step 4: AI nuanced review ─────────
        print("   🤖 Pass 2: AI compliance + brand voice review...")
        prompt = build_reviewer_prompt(
            draft_text       = draft_text,
            content_row      = content_row,
            research_result  = research_result,
            compliance_flags = compliance_scan["flags"],
            config           = config
        )

        try:
            raw_response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        ).text.strip()
        except Exception as e:
            print(f"🚨 Reviewer AI error: {e}")
            return {
                "status":                 "BLOCKED",
                "compliance_result":      "UNKNOWN",
                "brand_voice_result":     "UNKNOWN",
                "violations":             [],
                "brand_notes":            [],
                "required_fixes":         [],
                "suggested_improvements": [],
                "reviewer_summary":       f"AI review failed: {str(e)}",
                "structural_issues":      structure["issues"],
                "warnings":               validation["warnings"],
                "errors":                 [f"Gemini API error: {str(e)}"]
            }

        # ── Step 5: Parse AI output ───────────
        ai_result = parse_reviewer_output(raw_response)

        # ── Step 6: Merge all results ─────────
        # Structural issues elevate status if AI passed
        all_required_fixes = (
            ai_result.get("required_fixes", []) +
            structure["issues"]
        )
        all_brand_notes = (
            ai_result.get("brand_notes", []) +
            structure["notes"] +
            validation["warnings"]
        )

        # Final status: most severe wins
        final_status = ai_result.get("status", "PASS")
        if all_required_fixes:
            final_status = "FAIL"

        # Build summary for Google Sheet
        if final_status == "PASS":
            reviewer_summary = "Passed all compliance and brand checks."
        elif final_status == "PASS_WITH_NOTES":
            reviewer_summary = (
                f"Passed compliance. {len(all_brand_notes)} suggestion(s): "
                f"{all_brand_notes[0] if all_brand_notes else ''}"
            )
        else:
            reviewer_summary = (
                f"FAIL — {len(all_required_fixes)} required fix(es): "
                f"{all_required_fixes[0] if all_required_fixes else 'See review notes.'}"
            )

        # ── Step 7: Report ───────────────────
        print(f"✅ Review complete.")
        print(f"   Status      : {final_status}")
        print(f"   Compliance  : {ai_result.get('compliance_result', 'PASS')}")
        print(f"   Brand Voice : {ai_result.get('brand_voice_result', 'PASS')}")
        if all_required_fixes:
            for fix in all_required_fixes:
                print(f"   ❌ {fix}")
        if all_brand_notes:
            for note in all_brand_notes[:3]:
                print(f"   💡 {note}")

        return {
            "status":                 final_status,
            "compliance_result":      ai_result.get("compliance_result",      "PASS"),
            "brand_voice_result":     ai_result.get("brand_voice_result",     "PASS"),
            "violations":             ai_result.get("violations",             []),
            "brand_notes":            all_brand_notes,
            "required_fixes":         all_required_fixes,
            "suggested_improvements": ai_result.get("suggested_improvements", []),
            "reviewer_summary":       reviewer_summary,
            "structural_issues":      structure["issues"],
            "warnings":               validation["warnings"] + compliance_scan["flags"],
            "errors":                 []
        }
