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
# On retry: receives previous_draft + required_fixes from the Reviewer
# and rewrites completely — never copies the rejected draft.
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATOR
# ──────────────────────────────────────────────────────────────────────────────

def validate_writer_inputs(content_row, research_result, config):
    errors   = []
    warnings = []

    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")
    if not content_row.get("Keyword"):
        errors.append("Keyword is required in content_row.")
    if not content_row.get("Section"):
        errors.append("Section is required in content_row.")
    if not content_row.get("BlogType"):
        warnings.append(
            "BlogType missing from content_row — defaulting to 'educational'. "
            "Check that the Planner is writing BlogType to the Content_Plan sheet."
        )

    if not research_result or not isinstance(research_result, dict):
        errors.append(
            f"research_result must be a dict from ResearcherAgent. "
            f"Got: {type(research_result).__name__}"
        )
    else:
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
        facts = research_result.get("key_facts", [])
        if len(facts) < 3:
            errors.append(
                f"Research brief has only {len(facts)} facts (minimum 3 required)."
            )
        citations = research_result.get("citations", [])
        if len(citations) < 3:
            errors.append(
                f"Research brief has only {len(citations)} citations (minimum 3 required)."
            )

    brand = config.get("brand", {})
    if not brand.get("brand_voice_summary"):
        warnings.append("brand_voice_summary missing — Writer will use default tone.")
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
# ──────────────────────────────────────────────────────────────────────────────

def extract_research_context(research_result):
    facts = research_result.get("key_facts", [])
    facts_block = "\n".join([
        f"- [{f.get('confidence', 'medium').upper()}] {f.get('fact', '')} "
        f"(Source: {f.get('source_url', 'unverified')})"
        for f in facts
    ]) or "No facts provided."

    citations = research_result.get("citations", [])
    citations_block = "\n".join([
        f"- {c.get('title', 'Untitled')} ({c.get('year', 'unknown')}) "
        f"[{c.get('source_type', 'study')}]: {c.get('url', '')}"
        for c in citations
    ]) or "No citations provided."

    boundaries = research_result.get("claim_boundaries", [])
    boundaries_block = "\n".join([f"- {b}" for b in boundaries]) \
        or "No specific boundaries noted — apply general compliance rules."

    entities = research_result.get("semantic_entities", [])
    entities_text = ", ".join(entities) if entities else "general wellness terms"

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
# ──────────────────────────────────────────────────────────────────────────────

def check_compliance(html, config=None):
    brand = (config or {}).get("brand", {})

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(class_=["fda-disclaimer", "disclaimer"]):
        tag.decompose()
    for p in soup.find_all("p"):
        if "food and drug administration" in p.get_text().lower():
            p.decompose()

    clean_text = soup.get_text().lower()

    raw_products  = brand.get("product_names",  "")
    raw_brand     = brand.get("brand_name",     "")
    product_list  = [p.strip() for p in raw_products.split(",") if p.strip()]
    if raw_brand and raw_brand not in product_list:
        product_list.append(raw_brand)
    product_pattern = "|".join(product_list) if product_list else "this product|our supplement"

    disease_claim_patterns = [
        r"\b(treats?|treating|treated)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(cures?|cured|curing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(heals?|healed|healing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(reverses?|reversed|reversing)\s+(insomnia|depression|anxiety|diabetes|cancer|dementia|disease|disorder|condition)\b",
        r"\b(prevents?|prevented|preventing)\s+(disease|cancer|diabetes|depression|disorder|dementia)\b",
        r"\b(diagnoses|diagnosed|diagnosing)\b.{0,30}\b(our product|Glomend|supplement)\b",
        r"\b(eliminates?|eliminated)\b.{0,30}\b(disease|condition|disorder)\b",
        r"works like a prescription",
        r"clinically proven to (treat|cure|prevent)",
        r"guaranteed (results|to work)",
        rf"({product_pattern}).{{0,60}}(insomnia|depression|anxiety disorder|diabetes|cancer|dementia|diagnoses|can diagnose|will diagnose|helps diagnose)",
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
            print(f"   ⚠️  Compliance regex error (skipped): {e}")

    return {"passed": len(violations) == 0, "violations": violations}


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSER
# ──────────────────────────────────────────────────────────────────────────────

def parse_writer_output(raw_html, config, blog_type="educational"):
    clean = re.sub(r"^```html\s*", "", raw_html, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$",    "", clean)
    clean = clean.strip()

    soup       = BeautifulSoup(clean, "html.parser")
    clean_html = str(soup)
    word_count = len(soup.get_text().split())

    blog_types        = config.get("blog_types", {})
    bt_config         = blog_types.get(blog_type, {})
    includes_faq      = bt_config.get("includes_faq",        True)
    has_product_link  = bt_config.get("target_product_link", False)

    warnings = []

    if not soup.find(["h2", "h3"]):
        warnings.append("No H2/H3 headings found — rewrite with proper subheadings.")
    if "sources" not in soup.get_text().lower() \
            and "references" not in soup.get_text().lower():
        warnings.append("No Sources/References section found — add a sources list.")
    if not soup.find("p"):
        warnings.append("No paragraph tags found — HTML structure is malformed.")
    if not soup.find(class_="hook"):
        warnings.append("Opening hook paragraph missing — add <p class='hook'>.")
    if includes_faq and not soup.find(class_="faq"):
        warnings.append(
            f"FAQ section missing for blog type '{blog_type}' "
            f"— add <section class='faq'>."
        )
    if has_product_link and not soup.find(class_="actionable-cta"):
        warnings.append(
            f"Product CTA section missing for blog type '{blog_type}' "
            f"— add <section class='actionable-cta'>."
        )

    brand    = config.get("brand", {})
    word_min = int(brand.get("default_word_count_min", 800))
    word_max = int(brand.get("default_word_count_max", 1200))

    if word_count < word_min:
        warnings.append(
            f"Word count ({word_count}) is below minimum ({word_min}). "
            f"Expand body sections with more detail — do not add filler."
        )
    elif word_count > word_max:
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
        print("✍️  Initializing Writer Agent...")
        self.config = config or {"brand": {}, "system": {}, "agent_rules": {}}

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
    # HTML STRUCTURE BUILDER
    # ──────────────────────────────────────────────────────────────────────────

    def _build_html_structure(self, blog_type, includes_faq,
                               has_product_link, products, disclaimer,
                               actionable_title, actionable_instruction):

        faq_block = """
<section class="faq">
  <h2>Frequently Asked Questions</h2>
  <div class="faq-item">
    <h3>[Question directly related to this article's topic]</h3>
    <p>[Answer — 2 to 3 sentences. Sound like a person talking, not a textbook.]</p>
  </div>
  <div class="faq-item">
    <h3>[Second specific question from this article's keyword]</h3>
    <p>[Answer]</p>
  </div>
  <div class="faq-item">
    <h3>[Third question]</h3>
    <p>[Answer]</p>
  </div>
</section>""" if includes_faq else "<!-- No FAQ for this blog type -->"

        cta_block = f"""
<section class="actionable-cta">
  <h2>{actionable_title}</h2>
  <p>{actionable_instruction}</p>
  <p>→ <a href="/products/[product-handle]">[Product Name]</a></p>
</section>""" if has_product_link else "<!-- No product CTA for this blog type -->"

        sources_block = """
<section class="sources">
  <h2>Sources</h2>
  <ul>
    <li><a href="[URL]" target="_blank">[Title] ([Year])</a></li>
  </ul>
</section>"""

        disclaimer_block = f'<p class="fda-disclaimer">{disclaimer}</p>'

        if blog_type == "educational":
            return f"""
Output ONLY raw HTML. No markdown. No backticks. No <html> or <body> tags.

REQUIRED STRUCTURE (all sections mandatory):

1. <p class="hook">1–2 sentence relatable opening. Something she's experienced.</p>

2. <p class="answer-first">Direct answer to the title question in 40–60 words.</p>

3. Two or three body sections — each answers a sub-question:
   <h2>[Question she would actually Google about this topic]</h2>
   <p>2–3 sentences. Include <span class="citation">[Source, Year]</span> for every claim.</p>

4. <h2>What This Means for You</h2>
   <p>Warm, practical takeaway. What should she actually do with this information?</p>

5. {faq_block}

6. {sources_block}

7. {disclaimer_block}"""

        elif blog_type == "how-to guide":
            return f"""
Output ONLY raw HTML. No markdown. No backticks. No <html> or <body> tags.

REQUIRED STRUCTURE (all sections mandatory):

1. <p class="hook">1–2 sentence relatable opening. The problem she's trying to solve.</p>

2. <p class="answer-first">What she'll be able to do after reading this. 40–60 words.</p>

3. Numbered steps — use an ordered list:
   <h2>Step-by-Step: [Short title]</h2>
   <ol>
     <li><strong>[Step name]</strong> — 2–3 sentences explaining what to do and why it works.</li>
     <!-- 4–6 steps total -->
   </ol>

4. <h2>Tips to Make This Work Long-Term</h2>
   <p>1–2 paragraphs of practical sustainability advice.</p>

5. {faq_block}

6. {sources_block}

7. {disclaimer_block}"""

        elif blog_type == "buying guide":
            return f"""
Output ONLY raw HTML. No markdown. No backticks. No <html> or <body> tags.

REQUIRED STRUCTURE (all sections mandatory):

1. <p class="hook">1–2 sentence opening about the confusion/overwhelm of choosing.</p>

2. <p class="answer-first">What this guide will help her decide. 40–60 words.</p>

3. <h2>What to Look for in a [Product Category]</h2>
   <p>2–3 sentences covering key ingredients, certifications, or quality markers.
   Include <span class="citation">[Source, Year]</span> for any ingredient claims.</p>

4. <h2>What to Avoid</h2>
   <p>1–2 paragraphs on red flags (fillers, underdosed ingredients, misleading labels).</p>

5. <h2>Our Pick: [Product Name]</h2>
   <p>2–3 paragraphs explaining WHY this product fits the criteria above.
   The recommendation must feel earned — not promotional.
   Link the product name: <a href="/products/[product-handle]">[Product Name]</a></p>

6. {faq_block}

7. {cta_block}

8. {sources_block}

9. {disclaimer_block}"""

        elif blog_type == "customer story":
            return f"""
Output ONLY raw HTML. No markdown. No backticks. No <html> or <body> tags.

REQUIRED STRUCTURE (all sections mandatory):

1. <p class="hook">Open in the middle of the struggle — not at the beginning of the story.</p>

2. <h2>The Problem That Felt Impossible to Solve</h2>
   <p>2–3 paragraphs describing the challenge in her own words.
   Specific details — not generic wellness speak.
   First-person ("I") or close-third ("She") — stay consistent throughout.</p>

3. <h2>The Turning Point</h2>
   <p>What changed? What did she try? How did she find the solution?
   This section must feel organic — not like an ad.</p>

4. <h2>Where She Is Now</h2>
   <p>Specific outcomes. How does her daily life look different?
   Use <span class="citation">[Source, Year]</span> if citing any research
   that supports the outcomes described.</p>

5. {cta_block}

6. {sources_block}

7. {disclaimer_block}"""

        elif blog_type == "case study":
            return f"""
Output ONLY raw HTML. No markdown. No backticks. No <html> or <body> tags.

REQUIRED STRUCTURE (all sections mandatory):

1. <p class="hook">1–2 sentences framing what was studied and why it matters.</p>

2. <h2>Context: Why This Was Studied</h2>
   <p>Background on the research question. Who were the participants?
   What was the hypothesis? Keep this factual and grounded.
   Use <span class="citation">[Source, Year]</span> for all claims.</p>

3. <h2>Key Findings</h2>
   <p>What did the research show? Use specific numbers where available.
   Example: "72% of participants reported..." / "After 8 weeks..."</p>
   <ul>
     <li>[Finding 1 with citation]</li>
     <li>[Finding 2 with citation]</li>
   </ul>

4. <h2>What the Research Means for You</h2>
   <p>Translate the findings into practical meaning for a real woman.
   Acknowledge any limitations: "This was a small study..." / "More research is needed..."</p>

5. {sources_block}

6. {disclaimer_block}"""

        else:
            print(f"   ⚠️  Unknown blog_type '{blog_type}' — using educational structure.")
            return self._build_html_structure(
                "educational", includes_faq, has_product_link,
                products, disclaimer, actionable_title, actionable_instruction
            )


    # ──────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ──────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, content_row, research_ctx, config,
                      previous_draft="", required_fixes=None):

        brand = config.get("brand", {})

        title         = content_row.get("Title",        "")
        keyword       = content_row.get("Keyword",      "")
        summary       = content_row.get("Summary",      "")
        section       = content_row.get("Section",      "")
        blog_type     = content_row.get("BlogType",     "educational").strip().lower()
        topic_cluster = content_row.get("TopicCluster", "")

        word_min = brand.get("default_word_count_min", "800")
        word_max = brand.get("default_word_count_max", "1200")

        brand_name     = brand.get("brand_name",              "the brand")
        brand_voice    = brand.get("brand_voice_summary",     "Empathetic, direct, and empowering.")
        cta_style      = brand.get("cta_style",               "soft — educational, not pushy")
        tone           = brand.get("tone_formality",           "conversational")
        age_range      = brand.get("audience_age_range",       "38–55")
        gender         = brand.get("audience_gender",          "female")
        pain_points    = brand.get("audience_pain_points",     "fatigue, brain fog, weight gain")
        sophistication = brand.get("audience_sophistication",  "educated, reads labels")
        products       = brand.get("product_names",            "")

        writer_persona = brand.get("writer_persona", "")
        if not writer_persona:
            writer_persona = (
                f"A real {gender} in her {age_range.split('–')[0]}s "
                f"who personally deals with {pain_points} "
                f"and now writes honestly about it for others like her. "
                f"Voice: {brand_voice}"
            )

        disclaimer = brand.get(
            "disclaimer_text",
            "*These statements have not been evaluated by the Food and Drug Administration. "
            "This product is not intended to diagnose, treat, cure, or prevent any disease."
        )
        compliance   = brand.get("compliance_framework", "FDA")
        industry     = brand.get("industry",             "supplements")
        competitors  = brand.get("competitor_names",     "")
        avoid_topics = brand.get("avoid_topics",         "competitor names, political topics")

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

        blog_types        = config.get("blog_types", {})
        bt_config         = blog_types.get(blog_type, {})
        includes_faq      = bt_config.get("includes_faq",                 True)
        has_product_link  = bt_config.get("target_product_link",           False)
        actionable_title  = bt_config.get("actionable_section_title",      "Ready to Try It?")
        actionable_instr  = bt_config.get("actionable_section_instruction",
                                          f"Explore how {products} may support your goals.")

        faq_prompt_note = (
            "FAQ SECTION: Include 3 questions DIRECTLY related to this article's "
            "title and keyword. Do NOT write generic questions — "
            "each must be specific to THIS article's angle and audience."
        ) if includes_faq else (
            f"FAQ SECTION: Do NOT include a FAQ section. "
            f"Blog type '{blog_type}' does not use FAQs — "
            f"follow the HTML structure exactly as specified."
        )

        cta_prompt_note = (
            f"PRODUCT CTA: This article type requires a product recommendation section. "
            f"Title: '{actionable_title}'. "
            f"Instruction: {actionable_instr} "
            f"Products available: {products}. "
            f"The CTA must feel earned and helpful — never promotional or pushy."
        ) if has_product_link else (
            f"PRODUCT CTA: Do NOT include a product CTA section. "
            f"Blog type '{blog_type}' is purely informational."
        )

        html_structure = self._build_html_structure(
            blog_type              = blog_type,
            includes_faq           = includes_faq,
            has_product_link       = has_product_link,
            products               = products,
            disclaimer             = disclaimer,
            actionable_title       = actionable_title,
            actionable_instruction = actionable_instr,
        )

        cluster_context = (
            f"TOPIC CLUSTER: {topic_cluster} — stay focused on this specific angle."
            if topic_cluster else ""
        )

        # ✅ FIX: revision block — se activa en intentos 2+ del rewrite loop
        # El orchestrator pasa previous_draft + required_fixes del Reviewer
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
Keep the same topic, title, keyword, blog type, and HTML structure.
════════════════════════════════════════"""

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
BLOG TYPE          : {blog_type.upper()}
SECTION            : {section}
INTENT / SUMMARY   : {summary}
{cluster_context}

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
BLOG TYPE INSTRUCTIONS
════════════════════════════════════════
You are writing a {blog_type.upper()} article. Follow these type-specific rules:

- educational    → Explains WHY something happens. Informational intent. No selling.
- how-to guide   → Tells HOW to do something. Step-by-step. Actionable today.
- buying guide   → Helps the reader CHOOSE. Evidence-based recommendation. Soft CTA.
- customer story → Narrative of TRANSFORMATION. First-person or close-third. Emotional.
- case study     → DATA-DRIVEN outcomes. Specific numbers. Acknowledge limitations.

{faq_prompt_note}

{cta_prompt_note}


════════════════════════════════════════
WRITE LIKE A HUMAN — ANTI-AI-DETECTION RULES
════════════════════════════════════════
SENTENCE LENGTH — BURSTINESS IS MANDATORY:
- Mix extremely short sentences with longer ones in every section.
- Short sentences hit hard. They create rhythm. Use them often.
- Never write three sentences in a row that are the same length.

NATURAL VOICE — SOUND LIKE A REAL PERSON:
- Use contractions everywhere: "you're", "it's", "doesn't", "they're".
- Start sentences with "But", "And", "So", "Yet", "Because".
- Use em-dashes for natural interruptions — like this — mid-sentence.
- Ask the reader direct rhetorical questions. Sound familiar?

IMPERFECT LOGIC — HUMANS DON'T OVER-EXPLAIN:
- Don't tie every paragraph into a perfect conclusion.
- Acknowledge complexity: "The research is mixed on this one."
- Disagree sometimes: "The common advice is X — but that's not the full story."

SPECIFIC DETAILS — NOT GENERIC STATEMENTS:
- Use real numbers: "In one 2023 study of 412 women..." "In as little as 4–6 weeks..."
- Concrete and specific — never vague or abstract.

BANNED AI PATTERNS — NEVER USE THESE:
- Never start a paragraph with: "Furthermore", "Moreover", "In addition",
  "It is worth noting", "Notably", "Interestingly", "Certainly", "Absolutely".
- Never end a section with a summary sentence that repeats what was just said.
- Never use: "comprehensive", "crucial", "vital", "robust", "dive into",
  "delve", "leverage", "navigate", "unlock", "holistic", "empower".

TONAL VARIATION:
- Start warm, get direct in the middle, end encouraging.
- Let mild frustration come through: "And honestly? That's exhausting."


════════════════════════════════════════
COMPLIANCE ({compliance})
════════════════════════════════════════
- NEVER make disease claims (treat / cure / prevent / diagnose / reverse).
- ALLOWED: "supports", "may help", "research suggests", "can contribute to".
- Required hedging for all health claims: {hedge_words}
- NEVER mention competitors: {competitors}
- AVOID these topics entirely: {avoid_topics}
- EVERY objective health claim needs: <span class="citation">[Source, Year]</span>

BANNED WORDS / PHRASES: {all_banned}


════════════════════════════════════════
HTML STRUCTURE — follow exactly for blog type: {blog_type.upper()}
════════════════════════════════════════
{html_structure}


════════════════════════════════════════
WORD COUNT: {word_min}–{word_max} words
(body content only — disclaimer and sources do not count)
════════════════════════════════════════{revision_block}"""


    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
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

        Returns:
          {
            "status":     "Draft Complete" | "Needs Review" | "HARD_FAIL" | "BLOCKED"
            "html":       str
            "word_count": int
            "warnings":   list
            "violations": list  — only populated if HARD_FAIL
            "errors":     list  — only populated if BLOCKED
          }
        """
        cfg       = config if config is not None else self.config
        title     = content_row.get("Title",    "untitled")
        blog_type = content_row.get("BlogType", "educational").strip().lower()
        is_retry  = bool(previous_draft and required_fixes)

        print(f"\n✍️  Writer running for: {title}")
        print(f"   📝 Blog type : {blog_type}")
        if is_retry:
            print(f"   🔁 REVISION MODE — fixing {len(required_fixes)} issue(s)")

        # ── Step 1: Validate ──────────────────────────────────────────────────
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

        # ── Step 2: Extract research context ──────────────────────────────────
        research_ctx = extract_research_context(research_result)

        # ── Step 3: Build prompt ──────────────────────────────────────────────
        prompt = self._build_prompt(
            content_row    = content_row,
            research_ctx   = research_ctx,
            config         = cfg,
            previous_draft = previous_draft,
            required_fixes = required_fixes or []
        )

        # ── Step 4: Call Gemini ───────────────────────────────────────────────
        # ✅ FIX: contents como estructura explícita Content/Part
        # Igual que en planner.py — evita APIError [400] con prompts largos
        print(f"   🤖 Calling Gemini ({self.model_name})...")
        try:
            raw_html = self.client.models.generate_content(
                model    = self.model_name,
                contents = [{"role": "user", "parts": [{"text": str(prompt)}]}]
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
        compliance = check_compliance(raw_html, config=cfg)
        if not compliance["passed"]:
            print("   🚨 HARD FAIL: Disease claim language detected.")
            for v in compliance["violations"]:
                print(f"      ❌ {v}")
            return {
                "status":     "HARD_FAIL",
                "errors":     [],
                "warnings":   [],
                "html":       raw_html,
                "word_count": 0,
                "violations": compliance["violations"]
            }

        # ── Step 6: Parse + validate structure ────────────────────────────────
        parsed       = parse_writer_output(raw_html, cfg, blog_type=blog_type)
        all_warnings = validation["warnings"] + parsed["warnings"]

        # ── Step 7: Final status ──────────────────────────────────────────────
        status = "Draft Complete" if not all_warnings else "Needs Review"

        print(f"   ✅ Writing complete.")
        print(f"      Status     : {status}")
        print(f"      Blog type  : {blog_type}")
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
