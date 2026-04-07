import os
import re
import json
import requests
from google import genai
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
from datetime import date


# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate_optimizer_inputs(draft_text, content_row, reviewer_result, config):
    """
    Hard-gate validation that runs BEFORE any Gemini call or tool use.

    Rules:
      - Errors   → block the pipeline entirely (return BLOCKED)
      - Warnings → logged and passed through, but do NOT block publishing

    Returns:
        dict: { "valid": bool, "errors": [str], "warnings": [str] }
    """
    errors   = []
    warnings = []

    # ── 1. Reviewer compliance gate — HARD REQUIREMENT ────────────────────
    # The Optimizer only runs on drafts that passed Writer → Reviewer.
    # If the reviewer status is anything other than PASS/PASS_WITH_NOTES,
    # the draft should go back to the Writer, not forward to Optimizer.
    if not reviewer_result or not isinstance(reviewer_result, dict):
        errors.append(
            "reviewer_result is required and must be a dict from ReviewerAgent."
        )
    else:
        status = reviewer_result.get("status", "")
        if status not in ("PASS", "PASS_WITH_NOTES"):
            errors.append(
                f"BLOCKED: Draft has not passed compliance review. "
                f"Reviewer status: '{status}'. "
                f"Summary: {reviewer_result.get('reviewer_summary', 'none')}"
            )

    # ── 2. Draft must exist and be substantive ─────────────────────────────
    # Minimum 200 chars prevents empty or placeholder drafts from passing.
    if not draft_text or not isinstance(draft_text, str):
        errors.append("draft_text is required and must be a string.")
    elif len(draft_text.strip()) < 200:
        errors.append(
            f"Draft is too short ({len(draft_text.strip())} chars). "
            "Minimum 200 characters required."
        )

    # ── 3. Required content_row fields ────────────────────────────────────
    # Title, Keyword, and Section are needed to build metadata and find
    # the correct Shopify blog. Without them, publishing cannot proceed.
    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")
    if not content_row.get("Keyword"):
        errors.append("Keyword is required in content_row.")
    if not content_row.get("Section"):
        errors.append("Section is required in content_row.")

    # ── 4. Brand config technical constraints ──────────────────────────────
    # These values control meta tag character limits and slug formatting.
    # Missing them would produce malformed SEO metadata.
    brand = config.get("brand", {})
    if not brand.get("meta_title_max_chars"):
        errors.append("Config missing: 'meta_title_max_chars' in Config_Brand.")
    if not brand.get("meta_description_max_chars"):
        errors.append("Config missing: 'meta_description_max_chars' in Config_Brand.")
    if not brand.get("slug_rules"):
        errors.append("Config missing: 'slug_rules' in Config_Brand.")

    # ── 5. Internal links — WARNING ONLY ──────────────────────────────────
    # If no internal link inventory is available, the Optimizer will insert
    # HTML placeholder comments instead. The article still publishes.
    if not content_row.get("available_internal_links"):
        warnings.append(
            "Internal link inventory unavailable. "
            "Placeholders will be inserted. "
            "Resolve before marking 'Ready for Approval'."
        )

    # ── 6. Schema config — WARNING ONLY ───────────────────────────────────
    # If schema_type is not set in Config_Brand, no JSON-LD is generated.
    # The article still publishes — just without structured data.
    if not brand.get("schema_type"):
        warnings.append(
            "schema_type missing from Config_Brand. "
            "Schema will not be generated."
        )

    # ── 7. Inline images — WARNING ONLY ───────────────────────────────────
    # The Writer does NOT embed inline <img> tags — this is intentional.
    # The featured image is attached by the Publisher via Shopify GraphQL.
    # This warning is purely informational and must NEVER block publishing.
    if "<img" not in draft_text:
        warnings.append(
            "No inline images in draft — this is expected. "
            "Featured image is attached by Publisher via Shopify API."
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_optimizer_output(raw_output):
    """
    Parses Gemini's structured text response into a usable dict.

    Gemini returns three labeled blocks in this order:
      --- METADATA ---    → meta_title, meta_description, url_slug, word_count
      --- SCHEMA ---      → JSON-LD script blocks (optional)
      --- OPTIMIZED HTML--- → final article HTML

    Any <!-- NEEDS REVIEW: ... --> comments embedded in the output are
    extracted and added to the warnings list.

    Returns:
        dict: {
            meta_title, meta_description, url_slug, word_count,
            warnings, schema, html, status
        }
    """
    result = {
        "meta_title":       "",
        "meta_description": "",
        "url_slug":         "",
        "word_count":       0,
        "warnings":         [],
        "schema":           "",
        "html":             "",
        "status":           "Ready for Approval"
    }

    # ── Extract METADATA block ─────────────────────────────────────────────
    # Captures everything between --- METADATA --- and the next section marker.
    meta_match = re.search(
        r"--- METADATA ---\n(.*?)(?=--- SCHEMA ---|--- OPTIMIZED HTML ---)",
        raw_output, re.DOTALL
    )
    if meta_match:
        meta_text = meta_match.group(1)
        for field, key in [
            ("META TITLE",       "meta_title"),
            ("META DESCRIPTION", "meta_description"),
            ("URL SLUG",         "url_slug"),
            ("WORD COUNT",       "word_count"),
            ("WARNINGS",         "warnings"),
        ]:
            m = re.search(rf"{field}:\s*(.+)", meta_text)
            if m:
                val = m.group(1).strip()
                if key == "word_count":
                    # Strip any non-numeric characters (e.g., "~900 words" → 900)
                    result[key] = int(re.sub(r"[^\d]", "", val) or 0)
                elif key == "warnings":
                    # "none" means no warnings — otherwise split by comma
                    result[key] = [] if val.lower() == "none" else \
                        [w.strip() for w in val.split(",") if w.strip()]
                else:
                    result[key] = val

    # ── Extract SCHEMA block ───────────────────────────────────────────────
    # Everything between --- SCHEMA --- and --- OPTIMIZED HTML ---
    schema_match = re.search(
        r"--- SCHEMA ---\n(.*?)(?=--- OPTIMIZED HTML ---)",
        raw_output, re.DOTALL
    )
    if schema_match:
        result["schema"] = schema_match.group(1).strip()

    # ── Extract HTML block ─────────────────────────────────────────────────
    # Everything after --- OPTIMIZED HTML --- to end of response.
    # Strip any markdown fences Gemini might accidentally include.
    html_match = re.search(
        r"--- OPTIMIZED HTML ---\n(.*)",
        raw_output, re.DOTALL
    )
    if html_match:
        result["html"] = (
            html_match.group(1)
            .replace("```html", "")
            .replace("```", "")
            .strip()
        )

    # ── Extract inline NEEDS REVIEW comments ──────────────────────────────
    # Gemini embeds <!-- NEEDS REVIEW: reason --> comments in the HTML
    # when it finds something that requires human attention.
    needs_review = re.findall(r"<!-- NEEDS REVIEW: (.*?) -->", raw_output)
    if needs_review:
        result["warnings"].extend(needs_review)

    # ── HTML fallback ──────────────────────────────────────────────────────
    # If Gemini didn't use the expected block format, try to use the full
    # response as raw HTML (last resort — avoids empty publish).
    if not result["html"]:
        result["warnings"].append(
            "Could not extract OPTIMIZED HTML block from output. "
            "Used raw response as fallback — review manually."
        )
        clean = re.sub(r"```html|```", "", raw_output).strip()
        if "<" in clean:
            result["html"] = clean

    # ── Final status ───────────────────────────────────────────────────────
    # Set to "Needs Review" only if there are warnings AFTER parsing.
    # The Orchestrator overrides this with non_blocking_warnings logic.
    if result["warnings"]:
        result["status"] = "Needs Review"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_prompt(draft_text, content_row, product_list_text,
                            related_reading, internal_links, config,
                            has_schema_config, has_internal_links,input_word_count=800):
    """
    Builds the full SEO + AEO optimization prompt sent to Gemini.

    Input  : Markdown draft that has passed Writer → Reviewer.
    Output : Structured text with METADATA / SCHEMA / OPTIMIZED HTML blocks.

    Key design decisions:
      - Gemini converts Markdown → HTML (Writer outputs Markdown)
      - Featured image is NOT part of this prompt (Publisher handles it)
      - Inline <img> tags are NOT expected or required in the HTML output
      - Schema is only generated if schema_type is set in Config_Brand
      - Internal links use placeholders if inventory is unavailable
    """
    brand                = config.get("brand", {})
    primary_keyword      = content_row.get("Keyword",            "")
    secondary_keywords   = content_row.get("SecondaryKeywords",   "none provided")
    title                = content_row.get("Title",               "")
    section              = content_row.get("Section",             "")
    author_name          = brand.get("author_name",               "Glomend Team")
    brand_name           = brand.get("brand_name",                "Glomend")
    meta_title_limit     = brand.get("meta_title_max_chars",      "60")
    meta_desc_limit      = brand.get("meta_description_max_chars","155")
    slug_rules           = brand.get("slug_rules",
                           "lowercase, hyphens only, include primary keyword, max 6 words")
    schema_type          = brand.get("schema_type",               "BlogPosting")
    faq_required         = brand.get("faq_schema_required",       "yes")
    word_min             = brand.get("default_word_count_min",    "800")
    word_max             = brand.get("default_word_count_max",    "1200")
    publish_date         = content_row.get("ScheduledDate",       str(date.today()))

    # ── Brand color palette ────────────────────────────────────────────────
    # Applied inline via style= attributes so Shopify renders them correctly.
    h2_color        = brand.get("h2_color",              "#18655C")
    h3_color        = brand.get("h3_color",              "#121212")
    h4_color        = brand.get("h4_color",              "#121212")
    faq_title_color = brand.get("faq_title_color",       "#4B2E4A")
    fda_color       = brand.get("fda_disclaimer_color",  "#575654")
    body_color      = brand.get("body_text_color",       "#121212")

    # ── Internal links block ───────────────────────────────────────────────
    # If a link inventory exists (from Config_InternalLinks or similar),
    # pass it to Gemini. Otherwise, instruct it to use placeholder comments.
    if has_internal_links:
        internal_link_block = f"INTERNAL LINK INVENTORY:\n{internal_links}"
    else:
        internal_link_block = (
            "INTERNAL LINK INVENTORY: Not available.\n"
            "Insert exactly 2 placeholder comments where links would go:\n"
            "<!-- INTERNAL LINK PLACEHOLDER: [describe target topic] -->"
        )

    # ── Schema block ───────────────────────────────────────────────────────
    # BlogPosting + FAQPage JSON-LD. Only generated if schema_type is set
    # in Config_Brand. If not configured, Gemini adds a NEEDS REVIEW comment.
    if has_schema_config:
        schema_block = (
            f"Generate {schema_type} JSON-LD schema.\n"
            f"FAQPage schema required: {faq_required}\n"
            f"Include: headline, datePublished ({publish_date}), "
            f"dateModified ({publish_date}), "
            f"author (name: {author_name}), "
            f"publisher (name: {brand_name}), "
            f"description (use meta description).\n\n"
            f"BlogPosting format:\n"
            f'{{\n'
            f'  "@context": "https://schema.org",\n'
            f'  "@type": "BlogPosting",\n'
            f'  "headline": "[meta title]",\n'
            f'  "description": "[meta description]",\n'
            f'  "author": {{"@type": "Person", "name": "{author_name}"}},\n'
            f'  "publisher": {{"@type": "Organization", "name": "{brand_name}"}},\n'
            f'  "datePublished": "{publish_date}",\n'
            f'  "dateModified": "{publish_date}",\n'
            f'  "keywords": "{primary_keyword}, {secondary_keywords}"\n'
            f'}}\n\n'
            f"FAQPage format (generate from FAQ section in the article):\n"
            f'{{\n'
            f'  "@context": "https://schema.org",\n'
            f'  "@type": "FAQPage",\n'
            f'  "mainEntity": [\n'
            f'    {{\n'
            f'      "@type": "Question",\n'
            f'      "name": "[Question]",\n'
            f'      "acceptedAnswer": {{"@type": "Answer", "text": "[Answer]"}}\n'
            f'    }}\n'
            f'  ]\n'
            f'}}'
        )
    else:
        schema_block = (
            "Schema not configured.\n"
            "Do NOT generate schema.\n"
            "Add at top of HTML: <!-- NEEDS REVIEW: schema not generated -->"
        )

    return f"""You are a senior SEO and Answer Engine Optimization (AEO) specialist for {brand_name}.

You receive a Markdown draft that has passed compliance and brand review.
Your job: convert it to clean HTML and apply all SEO/AEO optimizations.

CRITICAL RULES — apply before any task:
1. Do NOT shorten, summarize, or remove any paragraph or sentence from the body.
2. Do NOT change any factual claims, health statements, or FDA disclaimers.
3. WORD COUNT: the input draft has ~{input_word_count} words of body text.
   After removing inline citations [Source, Year], the output must retain
   at least {int(input_word_count * 0.90)} words of visible text.
   If you are below this, you have cut content — add it back.

════════════════════════════════════════
INPUTS
════════════════════════════════════════
PRIMARY KEYWORD    : {primary_keyword}
SECONDARY KEYWORDS : {secondary_keywords}
SECTION            : {section}
TITLE              : {title}
AUTHOR             : {author_name}
PUBLISH DATE       : {publish_date}
WORD COUNT RANGE   : {word_min}–{word_max} words

AVAILABLE PRODUCTS TO LINK:
{product_list_text}

POTENTIAL RELATED ARTICLE:
{related_reading if related_reading else "None available."}

{internal_link_block}

MARKDOWN DRAFT (compliance-approved — convert to HTML, do NOT change content):
{draft_text}


════════════════════════════════════════
TASKS (complete all in order)
════════════════════════════════════════

─── TASK 1: MARKDOWN → HTML CONVERSION ───────────────────────────
Convert the Markdown draft to clean HTML using these rules:

Headings:
- ## Heading  → <h2 style="color: {h2_color};">[text]</h2>
- ### Heading → <h3 style="color: {h3_color};">[text]</h3>
- #### Heading→ <h4 style="color: {h4_color};">[text]</h4>

Inline formatting:
- **bold**   → <strong>
- *italic*   → <em>
- - list item → <li> inside <ul>
- Paragraph breaks → <p> tags with style="color: {body_color};"
  (EXCEPTION: hook, answer-first, fda-disclaimer, author-byline
   use their own class — do NOT add the body color style to those)

Citations:
- [Source, Year] inline citations → REMOVE from body text entirely.
  Sources are listed ONLY in the <section class="sources"> at the bottom.

Special elements:
- Opening hook paragraph     → <p class="hook">[text]</p>
- Answer-first paragraph     → <p class="answer-first">[text]</p>
- FDA disclaimer line        → <p class="fda-disclaimer"
                                  style="color: {fda_color}; font-style: italic;">[text]</p>
- FAQ section                → <section class="faq">
                                  <h2 style="color: {faq_title_color};">Frequently Asked Questions</h2>
                                  <div class="faq-item"><h3>[Q]</h3><p>[A]</p></div>
                               </section>
- Sources section            → <section class="sources">
                                  <h2>Sources</h2>
                                  <ul><li><a href="[URL]">[Title] ([Year])</a></li></ul>
                               </section>

Do NOT add <html>, <head>, or <body> wrapper tags.
Do NOT add any <img> tags — images are handled by the Publisher separately.


─── TASK 2: SEO STRUCTURE VERIFICATION ───────────────────────────
After conversion, verify and fix only if clearly broken:

2a. NO <h1> tag. The article title is rendered by Shopify separately.
    If an <h1> exists in the draft, REMOVE it entirely.
2b. First paragraph must include "{primary_keyword}" naturally.
    If missing, add it naturally — do NOT force it awkwardly.
2c. At least one <h2> must be phrased as a question.
2d. Secondary keywords ({secondary_keywords}) must appear in
    at least one <h2> or <p>.
2e. Do NOT add or remove heading levels beyond the rules above.


─── TASK 3: INTERNAL LINKING ─────────────────────────────────────
3a. Inject 1–2 product links from AVAILABLE PRODUCTS TO LINK
    within existing sentences — do NOT create new sentences for them.
    Format: <a href="URL">Product Name</a>

3b. If POTENTIAL RELATED ARTICLE is provided, append before the
    sources section:
    <div class="related-reading">
      <p><strong>Related Reading:</strong> <a href="[URL]">[Title]</a></p>
    </div>

3c. Add 2 internal links from INTERNAL LINK INVENTORY.
    If inventory is unavailable, insert placeholder comments:
    <!-- INTERNAL LINK PLACEHOLDER: [describe target topic] -->

3d. All anchor text must be descriptive.
    NEVER use "click here", "learn more", or bare URLs as anchor text.


─── TASK 4: AEO STRUCTURE ────────────────────────────────────────
AEO (Answer Engine Optimization) helps Perplexity, ChatGPT, and
Google AI Overviews extract and cite this content accurately.

4a. Verify opening paragraph answers the title question in 40–60 words.
    If it doesn't, add one answer sentence before the first paragraph:
    <!-- AEO: added answer-first sentence -->

4b. Verify ≥ 3 headings are phrased as questions.
    Rephrase existing headings if needed — do not add new ones.

4c. Verify FAQ section exists with 3–5 Q&A pairs.
    If missing, add it before the Sources section:
    <section class="faq">
      <h2 style="color: {faq_title_color};">Frequently Asked Questions</h2>
      <div class="faq-item">
        <h3>[Specific question a perimenopausal woman would actually Google]</h3>
        <p>[Direct 2–3 sentence answer — no filler, no hedging headers]</p>
      </div>
    </section>

4d. Verify Sources section exists.
    If missing, add:
    <section class="sources"><h2>Sources</h2></section>
    <!-- NEEDS REVIEW: sources section was empty — add citations manually -->


─── TASK 5: METADATA ─────────────────────────────────────────────
Generate SEO metadata for Shopify:

META TITLE:
  - Primary keyword near the start
  - Max {meta_title_limit} characters (hard limit — never exceed)
  - Format: [Primary keyword phrase] | {brand_name}

META DESCRIPTION:
  - Max {meta_desc_limit} characters (hard limit)
  - Must include the primary keyword
  - End with a clear value proposition
  - Zero disease-claim language

URL SLUG:
  - Rules: {slug_rules}
  - Must include: {primary_keyword}

Author byline — append before the closing of the article body:
<p class="author-byline">Written by <strong>{author_name}</strong> | Last Updated: {publish_date}</p>


─── TASK 6: SCHEMA ───────────────────────────────────────────────
{schema_block}


─── TASK 7: TECHNICAL VALIDATION ─────────────────────────────────
Run these checks on the final HTML before outputting:

7a. No broken href attributes (no empty href="" or href="#").
7b. No duplicate heading tags of the same level in the same section.
7c. IMPORTANT — DO NOT flag missing <img> tags as an error.
    The featured image is attached by the Shopify Publisher via GraphQL
    after optimization. Inline images are intentionally absent.
7d. All opened tags are properly closed.
7e. Word count {word_min}–{word_max}: if outside this range, add a warning
    in the WARNINGS field of METADATA — but DO NOT cut citations or FAQ.
7f. FDA disclaimer must not be modified in any way.


════════════════════════════════════════
OUTPUT FORMAT — return in EXACTLY this order, no extra text before or after
════════════════════════════════════════

--- METADATA ---
META TITLE: [value]
META DESCRIPTION: [value]
URL SLUG: [value]
WORD COUNT: [number]
WARNINGS: [comma-separated list or "none"]

--- SCHEMA ---
[JSON-LD <script> blocks — omit this entire section if schema is not configured]

--- OPTIMIZED HTML ---
[Full HTML output — raw only, no markdown fences, no backticks]


════════════════════════════════════════
ABSOLUTE CONSTRAINTS
════════════════════════════════════════
- DO NOT change any factual sentences, health claims, or citations.
- DO NOT add new statistics or claims not present in the original draft.
- DO NOT modify the FDA disclaimer text in any way.
- DO NOT add <img> tags — images are handled by the Publisher.
- DO NOT use markdown formatting anywhere in the HTML output.
- If unsure whether a change is needed → DO NOT make it."""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OptimizerAgent:

    def __init__(self, config=None):
        """
        Agent 5: SEO & AEO Optimizer.

        Responsibilities:
          1. Validate inputs (reviewer passed, draft exists, config complete)
          2. Fetch Shopify product links (for internal CTA injection)
          3. Fetch related articles from ChromaDB (for related reading section)
          4. Build and send prompt to Gemini
          5. Parse structured output into HTML + metadata + schema
          6. Classify warnings as blocking or non-blocking
          7. Return structured result dict to Orchestrator

        Tools:
          - Shopify Admin GraphQL API  → product link list
          - ChromaDB vector search     → related article suggestions
          - Gemini                     → Markdown → HTML + SEO optimization

        Image handling:
          - The Optimizer does NOT fetch or embed images.
          - Featured image is attached by PublisherAgent via Shopify GraphQL.
          - This is intentional: Optimizer only handles text content.
        """
        print("⚙️ Initializing Optimizer Agent...")
        self.config = config or {"brand": {}, "system": {}}

        # ── Gemini client ──────────────────────────────────────────────────
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

        # Model priority: Config_System sheet → env var → hardcoded default
        self.model_name = (
            self.config.get("system", {}).get("Gemini_Model")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        print(f"   🤖 Model: {self.model_name}")

        # ── Shopify connection ─────────────────────────────────────────────
        # Used only to fetch live product handles for link injection.
        # Falls back to hardcoded product list if credentials are missing.
        shop = os.getenv("SHOPIFY_SHOP")
        self.shopify_domain = f"{shop}.myshopify.com" if shop else ""
        self.shopify_token  = os.getenv("SHOPIFY_ACCESS_TOKEN")

        # ── ChromaDB vector store ──────────────────────────────────────────
        # Lazy-initialised on first use to avoid blocking startup on model download.
        self.chroma_client = None
        self.embed_fn      = None
        self.collection    = None
        self._chroma_ready = False

        # ── Non-blocking warning keywords ──────────────────────────────────
        # Warnings that contain any of these strings will NOT change the
        # article status to "Needs Review". They are informational only.
        # The Orchestrator has its own list from Agent_Rules — this is the
        # code-level default that applies even without a Sheet entry.
        self.non_blocking_keywords = [
            "Internal link inventory unavailable",  # expected — no link inventory set up yet
            "schema_type missing",                  # expected — schema config optional
            "No inline images",                     # ✅ FIX: images are handled by Publisher
            "Featured image is attached",           # ✅ FIX: informational only
            "Placeholders will be inserted",        # expected — internal links placeholder
            "Word count",
            "word count",
            "below the"
        ]


    # ── ChromaDB lazy init ───────────────────────────────────────────────
    def _init_chroma(self):
        """Load the SentenceTransformer model and connect to ChromaDB on first use."""
        if self._chroma_ready or not CHROMADB_AVAILABLE:
            return
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="blog_history",
                embedding_function=self.embed_fn
            )
            self._chroma_ready = True
        except Exception as e:
            print(f"⚠️  ChromaDB init failed — related-reading disabled: {e}")

    # ── TOOL 1: Shopify Products ─────────────────────────────────────────
    def get_shopify_products_graphql(self):
        """
        Fetches the list of active products from Shopify via GraphQL.

        Returns:
            list of dicts: [{"title": str, "url": str}, ...]

        Falls back to a hardcoded product list if:
          - SHOPIFY_SHOP or SHOPIFY_ACCESS_TOKEN env vars are missing
          - The API call fails for any reason

        The product list is injected into the Optimizer prompt so Gemini
        can link to real product pages within the article body.
        """
        fallback_products = [
            {"title": "GloRest",                        "url": "/products/glorest"},
            {"title": "GloSerene",                      "url": "/products/gloserene"},
            {"title": "GloBalance",                     "url": "/products/globalance"},
            {"title": "The Complete Day & Night System", "url": "/products/the-complete-day-night-system"}
        ]

        if not self.shopify_domain or not self.shopify_token:
            print("⚠️ Shopify credentials missing. Using fallback product links.")
            return fallback_products

        url     = f"https://{self.shopify_domain}/admin/api/2024-04/graphql.json"
        headers = {
            "X-Shopify-Access-Token": self.shopify_token,
            "Content-Type":           "application/json"
        }
        query = """
        {
          products(first: 20, query: "status:active") {
            edges { node { title handle } }
          }
        }
        """
        try:
            response = requests.post(
                url, json={"query": query}, headers=headers, timeout=10
            )
            if response.status_code == 200:
                edges = response.json()["data"]["products"]["edges"]
                return [
                    {"title": e["node"]["title"],
                     "url":   f"/products/{e['node']['handle']}"}
                    for e in edges
                ]
            print(f"⚠️ Shopify returned status {response.status_code}.")
            return fallback_products
        except Exception as e:
            print(f"⚠️ Shopify GraphQL failed: {e}")
            return fallback_products


    # ── TOOL 2: ChromaDB Related Articles ────────────────────────────────
    def get_related_blogs(self, current_summary):
        """
        Finds the most semantically similar published article using ChromaDB.

        Uses the current article's summary as the query vector.
        Returns the closest match as "Title|URL" string, or "" if none found.

        This result is injected into the prompt so Gemini can add a
        <div class="related-reading"> section at the end of the article.
        """
        self._init_chroma()
        if not self.collection:
            return ""
        try:
            results = self.collection.query(
                query_texts=[current_summary],
                n_results=1
            )
            if results["documents"] and len(results["documents"][0]) > 0:
                title = results["metadatas"][0][0]["title"]
                url   = results["metadatas"][0][0]["url"]
                return f"{title}|{url}"
        except Exception as e:
            print(f"⚠️ ChromaDB search failed: {e}")
        return ""


    # ── TOOL 3: Save to Vector Memory ────────────────────────────────────
    def add_to_memory(self, title, summary, url):
        """
        Saves a successfully published article to ChromaDB vector memory.

        Called by the PublisherAgent after a successful Shopify publish —
        NOT during the optimization step itself.

        Stored data is used by get_related_blogs() in future runs to
        suggest related reading links between articles.

        Args:
            title   : article title (used as document ID)
            summary : plain text summary (used as embedding vector)
            url     : Shopify storefront URL (e.g., /blogs/sleep/article-handle)
        """
        self._init_chroma()
        if not self.collection:
            return
        try:
            self.collection.add(
                documents=[summary],
                metadatas=[{"title": title, "url": url}],
                ids=[title.replace(" ", "_").lower()]
            )
            print(f"🧠 Added '{title}' to vector memory.")
        except Exception as e:
            print(f"🚨 Failed to add to vector memory: {e}")


    # ── MAIN: optimize_draft ─────────────────────────────────────────────
    def optimize_draft(self, content_row, draft_text, reviewer_result, config=None):
        """
        Full SEO & AEO optimization pipeline for one article.

        Flow:
          1. Validate inputs           → block if errors
          2. Fetch Shopify products    → for product link injection
          3. Fetch related article     → for related-reading section
          4. Build Gemini prompt       → includes all context and tasks
          5. Call Gemini               → get structured METADATA/SCHEMA/HTML
          6. Parse structured output   → extract each section
          7. Classify warnings         → blocking vs non-blocking
          8. Return result dict        → Orchestrator writes to Sheet

        Args:
            content_row     : dict from Content_Plan row
                              (Title, Keyword, Section, ScheduledDate,
                               SecondaryKeywords, available_internal_links)
            draft_text      : str — plain Markdown from Writer/Reviewer
            reviewer_result : dict — must have status PASS or PASS_WITH_NOTES
            config          : dict — full config from Google Sheets tabs

        Returns:
            dict: {
                status            : "Ready for Approval" | "Needs Review" | "BLOCKED"
                html              : str — final optimized HTML for Shopify
                meta_title        : str
                meta_description  : str
                url_slug          : str
                word_count        : int
                schema            : str — JSON-LD script blocks
                warnings          : [str]
                errors            : [str]
            }
        """
        if config is None:
            config = self.config

        title   = content_row.get("Title",   "untitled")
        summary = content_row.get("Summary", title)

        print(f"\n⚙️ Optimizer running for: {title}")

        # ── Step 1: Validate inputs ──────────────────────────────────────
        validation = validate_optimizer_inputs(
            draft_text, content_row, reviewer_result, config
        )
        if not validation["valid"]:
            print("🚨 Optimizer BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"   ❌ {err}")
            return {
                "status":           "BLOCKED",
                "errors":           validation["errors"],
                "warnings":         validation["warnings"],
                "html":             "",
                "meta_title":       "",
                "meta_description": "",
                "url_slug":         "",
                "word_count":       0,
                "schema":           ""
            }

        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"   ⚠ {w}")

        # ── Step 2: Fetch Shopify products ───────────────────────────────
        products = self.get_shopify_products_graphql()
        product_list_text = "\n".join([
            f"- {p['title']} (URL: {p['url']})" for p in products
        ])

        # ── Step 3: Fetch related article from ChromaDB ──────────────────
        # Returns "Title|URL" or "" if no related article exists yet.
        raw_related     = self.get_related_blogs(summary)
        related_reading = ""
        if raw_related and "|" in raw_related:
            parts = raw_related.split("|", 1)
            related_reading = (
                f"Title: {parts[0].strip()} | URL: {parts[1].strip()}"
            )

        has_internal_links = bool(content_row.get("available_internal_links"))
        has_schema_config  = bool(config.get("brand", {}).get("schema_type"))

        # ── Step 4: Build prompt ─────────────────────────────────────────
        input_word_count = len(re.sub(r'[^\\w\\s]', ' ', draft_text).split())
        prompt = build_optimizer_prompt(
            draft_text         = draft_text,
            content_row        = content_row,
            product_list_text  = product_list_text,
            related_reading    = related_reading,
            internal_links     = content_row.get("available_internal_links", ""),
            config             = config,
            has_schema_config  = has_schema_config,
            has_internal_links = has_internal_links
        )

        # ── Step 5: Call Gemini ──────────────────────────────────────────
        print(f"   🤖 Calling Gemini ({self.model_name})...")
        try:
            raw_response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt
            ).text.strip()
        except Exception as e:
            print(f"🚨 Gemini API error: {e}")
            return {
                "status":           "BLOCKED",
                "errors":           [f"Gemini API error: {str(e)}"],
                "warnings":         [],
                "html":             "",
                "meta_title":       "",
                "meta_description": "",
                "url_slug":         "",
                "word_count":       0,
                "schema":           ""
            }

        # ── Step 6: Parse structured output ─────────────────────────────
        parsed = parse_optimizer_output(raw_response)

        # Prepend validation warnings (e.g., "internal links unavailable")
        # so they appear at the top of the warning list in the Sheet.
        parsed["warnings"] = validation["warnings"] + parsed["warnings"]

        # ── Step 7: Classify warnings ────────────────────────────────────
        # Non-blocking = informational only → article still gets published.
        # Blocking     = something actually wrong → "Needs Review" in Sheet.
        #
        # self.non_blocking_keywords covers code-level defaults.
        # The Orchestrator applies an additional layer from Agent_Rules sheet.
        blocking_warnings = [
            w for w in parsed["warnings"]
            if not any(nb in w for nb in self.non_blocking_keywords)
        ]

        if blocking_warnings:
            parsed["status"] = "Needs Review"
        else:
            # ✅ "Ready for Approval" → Orchestrator maps this to "Ready to Publish"
            parsed["status"] = "Ready for Approval"

        # ── Step 8: Report ───────────────────────────────────────────────
        print(f"✅ Optimization complete.")
        print(f"   Status      : {parsed['status']}")
        print(f"   Meta Title  : {parsed['meta_title']}")
        print(f"   URL Slug    : {parsed['url_slug']}")
        print(f"   Word Count  : {parsed['word_count']}")
        if parsed["warnings"]:
            for w in parsed["warnings"]:
                print(f"   ⚠ {w}")

        # Always include "errors" key so Orchestrator never gets a KeyError
        parsed["errors"] = []
        return parsed
