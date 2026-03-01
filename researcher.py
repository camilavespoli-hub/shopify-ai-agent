import os
import re
import json
import requests
from google import genai
from ddgs import DDGS
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCHER AGENT — Agent 2 of the pipeline
#
# Responsibilities:
#   1. Gather live web data (DuckDuckGo multi-query search)
#   2. Fetch peer-reviewed citations from PubMed/NCBI API
#   3. Pull semantic entity context from Wikidata
#   4. Ask Gemini to synthesize everything into a structured Research Brief
#   5. Return the brief as a JSON dict — the Writer uses this to write the post
#
# All brand context (language, compliance, audience, sources) comes from
# Config_Brand and Agent_Rules — nothing is hardcoded.
# ══════════════════════════════════════════════════════════════════════════════

class ResearcherAgent:

    def __init__(self, config=None):
        """
        Initializes the Researcher Agent.
        Reads API keys, model name, and brand config from the Orchestrator.
        """
        print("🔬 Initializing Researcher Agent...")
        self.config = config or {"brand": {}, "system": {}, "agent_rules": {}}

        # ── Gemini client ─────────────────────────────────────────────────────
        # Model priority: Config_System tab → GEMINI_MODEL env var → hardcoded default
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("   ❌ GOOGLE_API_KEY missing — Researcher will fail.")

        self.client = genai.Client(api_key=api_key)

        self.model_name = (
            self.config.get("system", {}).get("Gemini_Model")
            or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        )
        print(f"   🤖 Model: {self.model_name}")

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 1: DuckDuckGo Multi-Query Web Search
    # Searches the live web for recent, relevant content on the topic.
    # We run multiple queries to maximize coverage:
    #   - one for clinical/biological context
    #   - one targeting preferred authority sources from Config_Brand
    #   - one for journal/academic content
    # ──────────────────────────────────────────────────────────────────────────

    def search_online_context(self, queries):
        """
        Runs multiple DuckDuckGo searches and combines all results into one string.

        queries : list of search query strings
        Returns : combined string of all search snippets, or a fallback message
        """
        all_results = []

        for query in queries:
            print(f"   🌐 Searching: '{query[:60]}...'")
            try:
                with DDGS() as ddgs:
                    results = [
                        f"[SOURCE] {r['title']}\n{r['body']}\nURL: {r['href']}"
                        for r in ddgs.text(query, max_results=3)
                    ]
                    all_results.extend(results)
            except Exception as e:
                # One failed query doesn't stop the others
                print(f"   ⚠️  DuckDuckGo search failed for query: {e}")

        return "\n\n".join(all_results) if all_results else "No live search results available."

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 2: PubMed / NCBI Academic Database
    # Fetches real peer-reviewed study citations for the keyword.
    # Uses the NCBI E-utilities API (free, no key needed).
    #
    # OLD: hardcoded "menopause" appended to every query — only worked for Glomend
    # NEW: appends the section name (e.g. "Sleep", "Mood & Stress") + industry
    #      so this works for any brand in any industry
    # ──────────────────────────────────────────────────────────────────────────

    def search_pubmed(self, keyword, section="", max_results=5):
        """
        Queries the PubMed database for peer-reviewed citations.

        keyword     : primary keyword for the post (e.g. "cortisol perimenopause")
        section     : blog section (e.g. "Sleep") — appended for relevance
        max_results : how many citations to retrieve

        Returns: list of dicts with title, URL, and year — or empty list on failure
        """
        brand    = self.config.get("brand", {})
        industry = brand.get("industry", "health")

        # Build the PubMed search term from keyword + section + industry
        # This replaces the hardcoded "menopause" with dynamic context
        search_term_parts = [keyword]
        if section:
            search_term_parts.append(section.lower())
        if industry and industry.lower() not in keyword.lower():
            search_term_parts.append(industry.lower())

        search_term = " ".join(search_term_parts)
        print(f"   📚 Querying PubMed: '{search_term}'...")

        # Step 1: Search for matching PubMed article IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db":      "pubmed",
            "term":    search_term,
            "retmax":  max_results,
            "retmode": "json",
            "sort":    "relevance"
        }

        try:
            res = requests.get(search_url, params=params, timeout=10).json()
            ids = res.get("esearchresult", {}).get("idlist", [])

            if not ids:
                print(f"   ℹ️  No PubMed results for '{search_term}'.")
                return []

            # Step 2: Fetch article summaries for the retrieved IDs
            sum_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            sum_res = requests.get(
                sum_url,
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                timeout=10
            ).json()

            # Step 3: Extract title, URL, and year for each article
            citations = []
            for pmid in ids:
                article = sum_res.get("result", {}).get(pmid, {})
                citations.append({
                    "title": article.get("title",   "Untitled Study"),
                    "url":   f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "year":  str(article.get("pubdate", ""))[:4] or "unknown"
                })

            print(f"   ✅ PubMed: {len(citations)} citation(s) found.")
            return citations

        except Exception as e:
            # PubMed failure is non-blocking — research continues without citations
            print(f"   ⚠️  PubMed query failed: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 3: Wikidata Semantic Entities
    # Gets related concepts and definitions to give Gemini richer context.
    # Example: for "cortisol" → returns ["steroid hormone", "stress response", ...]
    # This helps Gemini write more accurate, specific content.
    # ──────────────────────────────────────────────────────────────────────────

    def get_semantic_entities(self, topic):
        """
        Queries Wikidata for concepts related to the topic keyword.

        topic   : keyword string (e.g. "cortisol perimenopause")
        Returns : list of up to 5 concept descriptions, or ["wellness"] as fallback
        """
        # Use content_language for the Wikidata query (was hardcoded "en")
        language = self.config.get("brand", {}).get("content_language", "en")

        url = (
            f"https://www.wikidata.org/w/api.php"
            f"?action=wbsearchentities"
            f"&search={topic}"
            f"&language={language}"
            f"&format=json"
        )
        try:
            res = requests.get(url, timeout=5).json()
            entities = [
                item["description"]
                for item in res.get("search", [])
                if "description" in item
            ][:5]
            return entities if entities else ["wellness"]
        except Exception as e:
            print(f"   ⚠️  Wikidata query failed: {e}")
            return ["wellness"]

    # ──────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # Assembles the Gemini prompt using all gathered data + Config_Brand context.
    # Every variable comes from the config — nothing is hardcoded.
    # ──────────────────────────────────────────────────────────────────────────

    def _build_research_prompt(self, title, section, summary, keyword,
                                entities, pubmed_cites, live_data,
                                source_policy, config):
        """
        Builds the full Gemini prompt for research brief generation.

        All brand, audience, language, and compliance context is read from
        Config_Brand — so this prompt works for any brand/industry.
        """
        brand = config.get("brand", {})

        # ── Brand & audience context ──────────────────────────────────────────
        brand_name     = brand.get("brand_name",             "the brand")
        age_range      = brand.get("audience_age_range",     "38–55")
        gender         = brand.get("audience_gender",        "female")
        pain_points    = brand.get("audience_pain_points",   "fatigue, brain fog, weight gain")
        sophistication = brand.get("audience_sophistication","educated, reads labels")

        # ── Language ──────────────────────────────────────────────────────────
        language = brand.get("content_language", "en")
        language_names = {
            "en": "English", "es": "Spanish", "pt": "Portuguese",
            "fr": "French",  "de": "German",  "it": "Italian"
        }
        language_label = language_names.get(language.lower(), language)

        # ── Compliance ────────────────────────────────────────────────────────
        industry    = brand.get("industry",             "supplements")
        compliance  = brand.get("compliance_framework", "FDA")
        disclaimer  = brand.get("disclaimer_text",      "")

        # Read forbidden phrases from Agent_Rules → reviewer
        # (same rules the Reviewer uses — ensures the brief stays compliant)
        forbidden_phrases = (
            config.get("agent_rules", {})
            .get("reviewer", {})
            .get("forbidden_phrases", "cure, treat, fix, guarantee, prevent")
        )
        hedge_words = (
            config.get("agent_rules", {})
            .get("reviewer", {})
            .get("required_hedge_words", "may, can support, research suggests")
        )

        # ── Source authority instructions ─────────────────────────────────────
        # preferred_source_domains comes from Config_Brand
        # Example: "pubmed.gov, nih.gov, examine.com"
        preferred_domains  = brand.get("preferred_source_domains", "pubmed.ncbi.nlm.nih.gov, nih.gov")
        min_authority      = brand.get("min_source_authority",     "high")

        return f"""You are a Senior Medical Researcher working for {brand_name}.

════════════════════════════════════════
LANGUAGE
════════════════════════════════════════
Write ALL output in {language_label}.
All keys must remain in English (they are code identifiers).
All VALUES (facts, warnings, claim_boundaries) must be in {language_label}.

════════════════════════════════════════
BRAND & AUDIENCE CONTEXT
════════════════════════════════════════
Brand              : {brand_name}
Industry           : {industry}
Compliance         : {compliance}
Section            : {section}
Target audience    : {gender}, age {age_range}
Pain points        : {pain_points}
Sophistication     : {sophistication}

════════════════════════════════════════
RESEARCH INPUTS
════════════════════════════════════════
Topic              : {title}
Intent             : {summary}
Primary keyword    : {keyword}
Semantic entities  : {", ".join(entities)}
PubMed citations   : {json.dumps(pubmed_cites, ensure_ascii=False)}
Live web data      :
{live_data}

════════════════════════════════════════
SOURCE QUALITY RULES
════════════════════════════════════════
Policy             : {source_policy}
Preferred domains  : {preferred_domains}
Minimum authority  : {min_authority}
- Prioritize peer-reviewed studies and government health sources.
- Exclude blog posts, forums, and commercial sites as primary sources.
- If a claim lacks a quality source, mark its confidence as "low".

════════════════════════════════════════
COMPLIANCE RULES ({compliance})
════════════════════════════════════════
FORBIDDEN phrases (never include these in key_facts or summaries):
  {forbidden_phrases}

REQUIRED hedging language when making health claims:
  {hedge_words}

Disclaimer to include where appropriate:
  "{disclaimer}"

════════════════════════════════════════
TASK
════════════════════════════════════════
Synthesize the research inputs above into a structured brief
that the Writer Agent will use to create a blog post.

REQUIREMENTS:
- key_facts    : exactly 5 grounded, specific facts with source URLs
                 each fact must reference a biological mechanism or clinical finding
                 confidence must be "high", "medium", or "low"
- claim_boundaries : list what this content CANNOT claim under {compliance} rules
                     be specific — not generic disclaimers
- citations    : include ALL sources used (PubMed + web), with title/URL/year
- warnings     : list any scientific uncertainties, conflicting studies,
                 or outdated data found during research

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════
Return ONLY a valid JSON object inside a ```json block.
No text outside the block. No comments inside the JSON.

```json
{{
  "topic":            "{title}",
  "section_context":  "{section}",
  "key_facts": [
    {{
      "fact":       "...",
      "confidence": "high",
      "source_url": "https://..."
    }}
  ],
  "claim_boundaries": [
    "Do not claim this product treats or cures any disease."
  ],
  "citations": [
    {{
      "title": "...",
      "url":   "https://...",
      "year":  "2024"
    }}
  ],
  "semantic_entities": {json.dumps(entities, ensure_ascii=False)},
  "warnings": [
    "..."
  ]
}}
```"""

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # Called by the Orchestrator for every row with Status = "Pending Approval"
    # that doesn't yet have a Research_Brief.
    # ──────────────────────────────────────────────────────────────────────────

    def research_topic(self, content_row, valid_sections, source_policy):
        """
        Orchestrates the full research process for one blog post topic.

        Steps:
          1. Pre-validation (section check, title quality)
          2. Gather live web data (DuckDuckGo)
          3. Fetch PubMed citations
          4. Get semantic entities from Wikidata
          5. Ask Gemini to synthesize into a Research Brief
          6. Parse and return the brief as a Python dict

        Parameters:
          content_row    : one row dict from the Content_Plan Sheet
          valid_sections : list of valid section names (from Config_Sections)
          source_policy  : the Source_Policy string from Config_Brand

        Returns a dict with at minimum:
          {"status": str, "warnings": [], ...plus all research fields}
        """
        title   = content_row.get("Title",   "")
        keyword = content_row.get("Keyword", "")
        summary = content_row.get("Summary", "")
        section = content_row.get("Section", "General Wellness")

        print(f"\n🔎 Researching [{section}]: {title}")

        # ── 1. PRE-VALIDATION ─────────────────────────────────────────────────
        # Check section is valid — catches typos in the Content_Plan Sheet
        valid_lower = [s.lower() for s in valid_sections]
        if section.lower() not in valid_lower:
            print(f"   ⚠️  Invalid section: '{section}' — not in Config_Sections.")
            return {
                "status":   "Needs Review: insufficient sources",
                "warnings": [f"Invalid section '{section}'. Valid: {valid_sections}"]
            }

        # Reject vague titles early — they produce poor research results
        if len(title.split()) < 5 and "?" not in title:
            print(f"   ⚠️  Title too vague: '{title}'")
            return {
                "status":   "Needs Review: insufficient sources",
                "warnings": ["Title too vague — fewer than 5 words and no question mark."]
            }

        # ── 2. BUILD SEARCH QUERIES ───────────────────────────────────────────
        # Query 1: general biological/clinical context for the topic
        # Query 2: targets authority sources from Config_Brand (not hardcoded)
        # Query 3: academic journals relevant to the industry

        brand              = self.config.get("brand", {})
        preferred_domains  = brand.get("preferred_source_domains", "mayoclinic.org, nih.gov")
        industry           = brand.get("industry",  "health")
        target_country     = brand.get("target_country", "US")

        # Build a site: filter string for DuckDuckGo from preferred_source_domains
        # Example: "pubmed.gov, nih.gov" → "site:pubmed.gov OR site:nih.gov"
        site_filter = " OR ".join(
            f"site:{d.strip()}"
            for d in preferred_domains.split(",")
            if d.strip()
        ) if preferred_domains else ""

        queries = [
            f"{title} biological mechanisms clinical evidence",
            f"{keyword} {industry} {site_filter}".strip(),
            f"{keyword} peer-reviewed study journal {target_country}"
        ]

        # ── 3. GATHER DATA IN PARALLEL (sequential for simplicity) ───────────
        live_data     = self.search_online_context(queries)
        pubmed_cites  = self.search_pubmed(keyword, section=section)
        entities      = self.get_semantic_entities(keyword)

        # ── 4. BUILD GEMINI PROMPT ────────────────────────────────────────────
        prompt = self._build_research_prompt(
            title        = title,
            section      = section,
            summary      = summary,
            keyword      = keyword,
            entities     = entities,
            pubmed_cites = pubmed_cites,
            live_data    = live_data,
            source_policy= source_policy,
            config       = self.config
        )

        # ── 5. CALL GEMINI ────────────────────────────────────────────────────
        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt
            )
            raw_text = response.text

        except Exception as e:
            print(f"   🚨 Gemini API error: {e}")
            return {
                "status":   "Needs Review: insufficient sources",
                "warnings": [f"Gemini API error: {e}"]
            }

        # ── 6. PARSE JSON RESPONSE ────────────────────────────────────────────
        # The prompt asks for a ```json ... ``` block.
        # We extract only that block to avoid parsing stray explanatory text.
        try:
            json_block = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)

            if json_block:
                json_str = json_block.group(1).strip()
            else:
                # Fallback: maybe Gemini returned raw JSON without the code block
                json_str = raw_text.strip()

            parsed_result = json.loads(json_str)

            # Stamp the status so the Orchestrator knows it's ready for the Writer
            parsed_result["status"] = "Research Complete"

            # Ensure warnings is always a list — the Orchestrator iterates over it
            if "warnings" not in parsed_result or not isinstance(parsed_result["warnings"], list):
                parsed_result["warnings"] = []

            # Ensure citations is always a list — the Writer expects to iterate it
            if "citations" not in parsed_result or not isinstance(parsed_result["citations"], list):
                parsed_result["citations"] = pubmed_cites  # fall back to PubMed direct

            # Ensure key_facts is always a list
            if "key_facts" not in parsed_result or not isinstance(parsed_result["key_facts"], list):
                parsed_result["key_facts"] = []
                parsed_result["warnings"].append("Gemini returned no key_facts — Writer will have limited context.")

            # Warn if fewer than 2 high-confidence sources were found
            high_conf = [
                f for f in parsed_result.get("key_facts", [])
                if f.get("confidence") == "high"
            ]
            if len(high_conf) < 2:
                parsed_result["warnings"].append(
                    f"Only {len(high_conf)} high-confidence fact(s) found — "
                    "Writer should hedge claims carefully."
                )

            print(f"   ✅ Research complete: "
                  f"{len(parsed_result.get('key_facts', []))} facts, "
                  f"{len(parsed_result.get('citations', []))} citations.")

            return parsed_result

        except json.JSONDecodeError as e:
            # JSON parsing failed — save raw output so you can debug it
            print(f"   🚨 JSON parse error: {e}")
            return {
                "status":    "Needs Review: insufficient sources",
                "warnings":  [f"Failed to parse Gemini output as JSON: {e}"],
                # Save the raw text to the Sheet so you can inspect what went wrong
                "raw_brief": raw_text[:500] if raw_text else ""
            }

        except Exception as e:
            print(f"   🚨 Researcher error: {e}")
            return {
                "status":   "Needs Review: insufficient sources",
                "warnings": [str(e)]
            }
