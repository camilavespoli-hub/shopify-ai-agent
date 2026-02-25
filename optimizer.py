import os
import re
import json
import requests
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from datetime import date



# ─────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────


def validate_optimizer_inputs(draft_text, content_row, reviewer_result, config):
    """
    Hard-gate validation. All errors must be empty to proceed.
    Returns {"valid": bool, "errors": [], "warnings": []}
    """
    errors   = []
    warnings = []


    # 1. Reviewer compliance gate — HARD REQUIREMENT
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


    # 2. Draft must exist and be substantive
    if not draft_text or not isinstance(draft_text, str):
        errors.append("draft_text is required and must be a string.")
    elif len(draft_text.strip()) < 200:
        errors.append(
            f"Draft is too short ({len(draft_text.strip())} chars). "
            "Minimum 200 characters required."
        )


    # 3. Content row fields
    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")
    if not content_row.get("Keyword"):
        errors.append("Keyword is required in content_row.")
    if not content_row.get("Section"):
        errors.append("Section is required in content_row.")


    # 4. Technical constraints — stop if missing (malformed metadata risk)
    brand = config.get("brand", {})
    if not brand.get("meta_title_max_chars"):
        errors.append("Config missing: 'meta_title_max_chars' in Config_Brand.")
    if not brand.get("meta_description_max_chars"):
        errors.append("Config missing: 'meta_description_max_chars' in Config_Brand.")
    if not brand.get("slug_rules"):
        errors.append("Config missing: 'slug_rules' in Config_Brand.")


    # 5. Internal links — warn only (can use placeholders)
    if not content_row.get("available_internal_links"):
        warnings.append(
            "NEEDS REVIEW: Internal link inventory unavailable. "
            "Placeholders will be inserted. "
            "Resolve before marking 'Ready for Approval'."
        )


    # 6. Schema config — warn only (graceful degradation)
    if not brand.get("schema_type"):
        warnings.append(
            "NEEDS REVIEW: schema_type missing from Config_Brand. "
            "Schema will not be generated."
        )


    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}



# ─────────────────────────────────────────────
# OUTPUT PARSER
# ─────────────────────────────────────────────


def parse_optimizer_output(raw_output):
    """
    Parses Gemini's structured output into usable components.
    Returns dict: meta_title, meta_description, url_slug,
                  word_count, warnings, schema, html, status
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


    # ── Extract METADATA block ────────────────
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
                    result[key] = int(re.sub(r"[^\d]", "", val) or 0)
                elif key == "warnings":
                    result[key] = [] if val.lower() == "none" else \
                        [w.strip() for w in val.split(",") if w.strip()]
                else:
                    result[key] = val


    # ── Extract SCHEMA block ──────────────────
    schema_match = re.search(
        r"--- SCHEMA ---\n(.*?)(?=--- OPTIMIZED HTML ---)",
        raw_output, re.DOTALL
    )
    if schema_match:
        result["schema"] = schema_match.group(1).strip()


    # ── Extract HTML block ────────────────────
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


    # ── Detect NEEDS REVIEW flags ─────────────
    needs_review = re.findall(r"<!-- NEEDS REVIEW: (.*?) -->", raw_output)
    if needs_review:
        result["warnings"].extend(needs_review)


    # ── If HTML block empty, fallback ─────────
    if not result["html"]:
        result["warnings"].append(
            "NEEDS REVIEW: Could not extract OPTIMIZED HTML block from output."
        )
        # Attempt to use full output as HTML fallback
        clean = re.sub(r"```html|```", "", raw_output).strip()
        if "<" in clean:
            result["html"] = clean


    # ── Final status ──────────────────────────
    if result["warnings"]:
        result["status"] = "Needs Review"


    return result



# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────


def build_optimizer_prompt(draft_text, content_row, product_list_text,
                            related_reading, internal_links, config,
                            has_schema_config, has_internal_links):
    """
    Builds the full SEO & AEO optimization prompt.
    Input: Markdown text from Reviewer.
    Output: structured block with METADATA / SCHEMA / OPTIMIZED HTML.
    """
    brand                = config.get("brand", {})
    primary_keyword      = content_row.get("Keyword",          "")
    secondary_keywords   = content_row.get("SecondaryKeywords", "none provided")
    title                = content_row.get("Title",            "")
    section              = content_row.get("Section",          "")
    author_name          = brand.get("author_name",             "Glomend Team")
    brand_name           = brand.get("brand_name",              "Glomend")
    meta_title_limit     = brand.get("meta_title_max_chars",    "60")
    meta_desc_limit      = brand.get("meta_description_max_chars", "155")
    slug_rules           = brand.get("slug_rules",
                        "lowercase, hyphens only, include primary keyword, max 6 words")
    schema_type          = brand.get("schema_type",             "BlogPosting")
    faq_required         = brand.get("faq_schema_required",     "yes")
    # ✅ FIX 3: defaults alineados con el Reviewer (800/1500, no 300/500)
    word_min             = brand.get("default_word_count_min",  "800")
    word_max             = brand.get("default_word_count_max",  "1500")
    publish_date         = content_row.get("ScheduledDate",     str(date.today()))
    h2_color             = brand.get("h2_color",                "#18655C")
    h3_color             = brand.get("h3_color",                "#121212")
    h4_color             = brand.get("h4_color",                "#121212")
    faq_title_color      = brand.get("faq_title_color",         "#4B2E4A")
    fda_color            = brand.get("fda_disclaimer_color",    "#575654")
    body_color           = brand.get("body_text_color",         "#121212")


    # Internal link block
    if has_internal_links:
        internal_link_block = f"INTERNAL LINK INVENTORY:\n{internal_links}"
    else:
        internal_link_block = (
            "INTERNAL LINK INVENTORY: Not available.\n"
            "Insert exactly 2 placeholder comments where links would go:\n"
            "<!-- INTERNAL LINK PLACEHOLDER: [describe target topic] -->"
        )


    # Schema block
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
            f"FAQPage format (generate from FAQ section):\n"
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


CRITICAL: Do NOT change any factual claims, citations, health statements, or FDA disclaimers.


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
Convert the Markdown draft to clean HTML:
- ## Heading → <h2 style="color: {h2_color};">[text]</h2>
- ### Heading → <h3 style="color: {h3_color};">[text]</h3>
- #### Heading → <h4 style="color: {h4_color};">[text]</h4>
- **bold** → <strong>
- *italic* → <em>
- - list item → <li> inside <ul>
- Paragraph breaks → <p> tags
- [Source, Year] citations → REMOVE from body text entirely.
Sources are listed only in the <section class="sources"> at the bottom.
- FDA disclaimer line → <p class="fda-disclaimer" style="color: {fda_color}; font-style: italic;">[text]</p>
- Body text: wrap all <p> tags (except hook, answer-first, fda-disclaimer, author-byline) with style="color: {body_color};"
- FAQ section → <section class="faq"><h2 style="color: {faq_title_color};">Frequently Asked Questions</h2>...</section>
- Opening hook paragraph → <p class="hook">[text]</p>
- Answer-first paragraph → <p class="answer-first">[text]</p>
- <section class="faq"><h2 style="color: #4B2E4A;">Frequently Asked Questions</h2>..</section>
Each Q → <div class="faq-item"><h3>[Q]</h3><p>[A]</p></div>
- Sources section → <section class="sources"><h2>Sources</h2><ul>...</ul></section>
Each source → <li><a href="[URL]">[Title] ([Year])</a></li>
Do NOT add <html>, <head>, or <body> tags.


─── TASK 2: SEO STRUCTURE VERIFICATION ───────────────────────────
After conversion, check and fix only if clearly broken:
2a. DO NOT add an <h1> tag. The title is published separately by Shopify.
    If an <h1> exists in the draft, REMOVE it entirely.
2b. First paragraph must include "{primary_keyword}" naturally.
2c. At least one <h2> must be phrased as a question.
2d. Secondary keywords ({secondary_keywords}) in at least one <h2> or <p>.
2e. Do NOT add or remove heading levels beyond the above.


─── TASK 3: INTERNAL LINKING ─────────────────────────────────────
3a. Inject 1–2 product links from AVAILABLE PRODUCTS TO LINK:
    <a href="URL">Product Name</a> — within existing sentences only.
3b. If POTENTIAL RELATED ARTICLE exists, append before closing:
    <div class="related-reading">
    <p><strong>Related Reading:</strong> <a href="[URL]">[Title]</a></p>
    </div>
3c. Add 2 internal links from INTERNAL LINK INVENTORY.
    If unavailable, insert placeholder comments.
3d. All anchor text must be descriptive — no "click here" or bare URLs.


─── TASK 4: AEO STRUCTURE ────────────────────────────────────────
4a. Verify opening paragraph answers the title question in 40–60 words.
    If not, add one answer sentence before first paragraph:
    <!-- AEO: added answer-first sentence -->
4b. Verify ≥3 headings are phrased as questions — rephrase existing ones if needed.
4c. Verify FAQ section exists with 3–5 Q&A pairs.
    If missing, add before Sources:
    <section class="faq">
    <h2 style="color: {faq_title_color};">Frequently Asked Questions</h2>
    <div class="faq-item">
        <h3>[Question a perimenopausal woman would actually ask]</h3>
        <p>[Direct answer, 2–3 sentences]</p>
    </div>
    </section>
4d. Verify Sources section exists. If missing, add:
    <section class="sources"><h2>Sources</h2></section>
    <!-- NEEDS REVIEW: sources section was empty -->


─── TASK 5: METADATA ─────────────────────────────────────────────
META TITLE:
  - Primary keyword near start
  - Max {meta_title_limit} characters
  - Format: [Primary keyword phrase] | {brand_name}


META DESCRIPTION:
  - Max {meta_desc_limit} characters
  - Include primary keyword
  - End with value proposition, no disease language


URL SLUG:
  - Rules: {slug_rules}
  - Must include: {primary_keyword}


Append author byline before closing:
<p class="author-byline">Written by <strong>{author_name}</strong> | Last Updated: {publish_date}</p>


─── TASK 6: SCHEMA ───────────────────────────────────────────────
{schema_block}


─── TASK 7: TECHNICAL VALIDATION ─────────────────────────────────
7a. No broken href attributes.
7b. No duplicate <h1> tags.
7c. All <img> tags have alt text with "{primary_keyword}".
7d. All opened tags are properly closed.
7e. Word count {word_min}–{word_max}: flag if outside range.
7f. FDA disclaimer must not be modified.


════════════════════════════════════════
OUTPUT FORMAT (return in EXACTLY this order)
════════════════════════════════════════


--- METADATA ---
META TITLE: [value]
META DESCRIPTION: [value]
URL SLUG: [value]
WORD COUNT: [number]
WARNINGS: [comma-separated list or "none"]


--- SCHEMA ---
[JSON-LD <script> blocks — omit entire section if schema not configured]


--- OPTIMIZED HTML ---
[Full HTML — raw only, no markdown fences, no backticks]


════════════════════════════════════════
ABSOLUTE CONSTRAINTS
════════════════════════════════════════
- DO NOT change any factual sentences, health claims, or citations.
- DO NOT add new claims or statistics.
- DO NOT modify the FDA disclaimer.
- DO NOT use markdown formatting in the HTML output.
- If unsure whether a change is needed, DO NOT make it."""



# ─────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────


class OptimizerAgent:
    def __init__(self, config=None):
        """
        Agent 5: SEO & AEO Optimizer.


        Input : plain Markdown text from ReviewerAgent
        Output: structured dict with HTML, metadata, schema, status


        Tools : Shopify GraphQL (product links), ChromaDB (related articles)
        """
        print("⚙️ Initializing Optimizer Agent...")
        self.config = config or {"brand": {}, "system": {}}


        # ── Gemini ──────────────────────────────
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        # ✅ FIX 1: model priority — Config_System → env var → default
        self.model_name = (
            self.config.get("system", {}).get("Gemini_Model")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        print(f"   🤖 Model: {self.model_name}")


        # ── Shopify ─────────────────────────────
        self.shopify_domain = os.getenv("SHOPIFY_STORE_DOMAIN")
        self.shopify_token  = os.getenv("SHOPIFY_ACCESS_TOKEN")


        # ── ChromaDB ────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="blog_history",
            embedding_function=self.embed_fn
        )


    # ── TOOL 1: Shopify Products ─────────────────────────────────────────
    def get_shopify_products_graphql(self):
        """TOOL: Fetches active products via Shopify Admin GraphQL API."""
        fallback_products = [
            {"title": "GloRest",                         "url": "/products/glorest"},
            {"title": "GloSerene",                       "url": "/products/gloserene"},
            {"title": "GloBalance",                      "url": "/products/globalance"},
            {"title": "The Complete Day & Night System",  "url": "/products/the-complete-day-night-system"}
        ]


        if not self.shopify_domain or not self.shopify_token:
            print("⚠️ Shopify credentials missing. Using fallback product links.")
            return fallback_products


        url = f"https://{self.shopify_domain}/admin/api/2024-04/graphql.json"
        headers = {
            "X-Shopify-Access-Token": self.shopify_token,
            "Content-Type": "application/json"
        }
        query = """
        {
          products(first: 20, query: "status:active") {
            edges { node { title handle } }
          }
        }
        """
        try:
            response = requests.post(url, json={"query": query},
                                     headers=headers, timeout=10)
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


    # ── TOOL 2: ChromaDB Related Blogs ───────────────────────────────────
    def get_related_blogs(self, current_summary):
        """TOOL: Semantic search in ChromaDB for related past articles."""
        try:
            results = self.collection.query(
                query_texts=[current_summary],
                n_results=1
            )
            if results["documents"] and len(results["documents"][0]) > 0:
                title = results["metadatas"][0][0]["title"]
                url   = results["metadatas"][0][0]["url"]
                # Return plain text — Optimizer prompt handles HTML formatting
                return f"{title}|{url}"
        except Exception as e:
            print(f"⚠️ ChromaDB search failed: {e}")
        return ""


    # ── TOOL 3: Save to Memory ────────────────────────────────────────────
    def add_to_memory(self, title, summary, url):
        """
        TOOL: Save a published post to ChromaDB.
        Called by Publisher after successful publish — not during optimization.
        """
        try:
            self.collection.add(
                documents=[summary],
                metadatas=[{"title": title, "url": url}],
                ids=[title.replace(" ", "_").lower()]
            )
            print(f"🧠 Added '{title}' to vector memory.")
        except Exception as e:
            print(f"🚨 Failed to add to vector memory: {e}")


    # ── MAIN: optimize_draft ──────────────────────────────────────────────
    def optimize_draft(self, content_row, draft_text, reviewer_result, config=None):
        """
        Full SEO & AEO optimization pipeline.


        Args:
            content_row     : dict — from Content_Plan
                              (Title, Keyword, Section, ScheduledDate,
                               SecondaryKeywords, available_internal_links)
            draft_text      : str  — plain Markdown from ReviewerAgent
            reviewer_result : dict — must contain {"status": "PASS" or
                              "PASS_WITH_NOTES", "reviewer_summary": "..."}
            config          : dict — from Google Sheets config tabs


        Returns:
            dict — status, html, meta_title, meta_description,
                   url_slug, word_count, schema, warnings, errors
        """
        if config is None:
            config = self.config


        title   = content_row.get("Title",   "untitled")
        summary = content_row.get("Summary", title)


        print(f"\n⚙️ Optimizer running for: {title}")


        # ── Step 1: Validate inputs ──────────
        validation = validate_optimizer_inputs(
            draft_text, content_row, reviewer_result, config
        )
        if not validation["valid"]:
            print("🚨 Optimizer BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"   ❌ {err}")
            return {
                "status":            "BLOCKED",
                "errors":            validation["errors"],
                "warnings":          validation["warnings"],
                "html":              "",
                "meta_title":        "",
                "meta_description":  "",
                "url_slug":          "",
                "word_count":        0,
                "schema":            ""
            }


        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"   ⚠ {w}")


        # ── Step 2: Gather tool outputs ──────
        products = self.get_shopify_products_graphql()
        product_list_text = "\n".join([
            f"- {p['title']} (URL: {p['url']})" for p in products
        ])


        # ChromaDB returns "Title|URL" or ""
        raw_related     = self.get_related_blogs(summary)
        related_reading = ""
        if raw_related and "|" in raw_related:
            parts = raw_related.split("|", 1)
            related_reading = (
                f"Title: {parts[0].strip()} | URL: {parts[1].strip()}"
            )


        has_internal_links = bool(content_row.get("available_internal_links"))
        has_schema_config  = bool(config.get("brand", {}).get("schema_type"))


        # ── Step 3: Build prompt ─────────────
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


        # ── Step 4: Call Gemini ──────────────
        # ✅ FIX 2: uses self.model_name — no hardcoded default
        print(f"   🤖 Calling Gemini ({self.model_name})...")
        try:
            raw_response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
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


        # ── Step 5: Parse structured output ──
        parsed = parse_optimizer_output(raw_response)


        # Merge input validation warnings
        parsed["warnings"] = validation["warnings"] + parsed["warnings"]


        # Non-blocking warnings — informational only, don't change status
        non_blocking = [
            "Internal link inventory unavailable",
            "schema_type missing",
        ]
        blocking_warnings = [
            w for w in parsed["warnings"]
            if not any(nb in w for nb in non_blocking)
        ]
        if blocking_warnings:
            parsed["status"] = "Needs Review"
        else:
            parsed["status"] = "Ready for Approval"


        # ── Step 6: Report ───────────────────
        print(f"✅ Optimization complete.")
        print(f"   Status      : {parsed['status']}")
        print(f"   Meta Title  : {parsed['meta_title']}")
        print(f"   URL Slug    : {parsed['url_slug']}")
        print(f"   Word Count  : {parsed['word_count']}")
        if parsed["warnings"]:
            for w in parsed["warnings"]:
                print(f"   ⚠ {w}")


        # ✅ FIX 4: always include "errors" key so Orchestrator never gets KeyError
        parsed["errors"] = []
        return parsed
