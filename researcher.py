import os
import re
import json
import requests
from google import genai
from ddgs import DDGS
from datetime import datetime

class ResearcherAgent:
    def __init__(self, config=None):
        """
        Agent 2: Medical Researcher.
        Upgraded with PubMed API, Multi-query Search, and JSON Handoff.
        """
        print("🔬 Initializing Researcher Agent (Elite Version)...")
        self.config = config or {"brand": {}}
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    # --- TOOL: DuckDuckGo Multi-Query ---
    def search_online_context(self, queries):
        all_results = []
        for query in queries:
            print(f"    🌐 Searching: '{query[:50]}...'")
            try:
                with DDGS() as ddgs:
                    results = [f"[SOURCE] {r['title']}\n{r['body']}\nURL: {r['href']}" for r in ddgs.text(query, max_results=3)]
                    all_results.extend(results)
            except Exception as e:
                print(f"    ⚠️ DDG failed: {e}")
        return "\n\n".join(all_results) if all_results else "No live search results."

    # --- TOOL: PubMed/NCBI API ---
    def search_pubmed(self, keyword, max_results=5):
        print(f"    📚 Querying PubMed for: '{keyword}'...")
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": f"{keyword} menopause", "retmax": max_results, "retmode": "json", "sort": "relevance"}
        
        try:
            res = requests.get(search_url, params=params, timeout=10).json()
            ids = res.get("esearchresult", {}).get("idlist", [])
            if not ids: return []

            sum_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            sum_res = requests.get(sum_url, params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}).json()
            
            citations = []
            for pmid in ids:
                article = sum_res.get("result", {}).get(pmid, {})
                citations.append({
                    "title": article.get("title", "Untitled Study"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "year": str(article.get("pubdate", ""))[:4] or "unknown"
                })
            return citations
        except: return []

    # --- TOOL: Semantic Entities ---
    def get_semantic_entities(self, topic):
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={topic}&language=en&format=json"
        try:
            res = requests.get(url, timeout=5).json()
            return [item['description'] for item in res.get('search', []) if 'description' in item][:5]
        except: return ["wellness"]

    # --- MAIN RESEARCH FUNCTION ---
    def research_topic(self, content_row, valid_sections, source_policy):
        title = content_row.get("Title", "")
        keyword = content_row.get("Keyword", "")
        summary = content_row.get("Summary", "")
        section = content_row.get("Section", "General Wellness")
        
        print(f"🔎 Researching [{section}]: {title}...")

        # --- PRE-RESEARCH VALIDATION ---
        valid_lower = [s.lower() for s in valid_sections]
        if section.lower() not in valid_lower:
            print(f"⚠️ Invalid section: '{section}'.")
            return {"status": "Needs Review: insufficient sources", "warnings": [f"Invalid section: {section}"]}

        if len(title.split()) < 5 and "?" not in title:
            print(f"⚠️ Title too vague.")
            return {"status": "Needs Review: insufficient sources", "warnings": ["Title too vague."]}

        # --- DATA GATHERING ---
        queries = [
            f"{title} biological mechanisms medical studies",
            f"{keyword} perimenopause symptoms site:mayoclinic.org OR site:acog.org",
            f"{keyword} scientific journal articles site:menopause.org OR site:tandfonline.com"
        ]
        
        live_data = self.search_online_context(queries)
        pubmed_cites = self.search_pubmed(keyword)
        entities = self.get_semantic_entities(keyword)

        # --- PROMPT BUILDER ---
        prompt = f"""
        Act as a Senior Medical Researcher for Glomend. 
        Your goal is to provide a targeted brief for the Section: {section}.
        
        INPUTS:
        - Topic: {title}
        - Intent: {summary}
        - Entities: {", ".join(entities)}
        - PubMed Data: {json.dumps(pubmed_cites)}
        - Web Data: {live_data}
        - Policy: {source_policy}

        TASK:
        Return ONLY a valid JSON object inside a ```json block. 
        
        STRUCTURE REQUIREMENTS:
        - key_facts: 5 grounded facts.
        - claim_boundaries: List exactly what we CANNOT say (e.g., 'Do not claim X treats disease Y').
        - citations: Include URLs and source types.
        - warnings: List any scientific uncertainties or outdated data found.

        FORMAT:
        ```json
        {{
          "topic": "{title}",
          "section_context": "{section}",
          "key_facts": [ {{"fact": "...", "confidence": "high", "source_url": "..."}} ],
          "claim_boundaries": [ "..." ],
          "citations": [ {{"title": "...", "url": "...", "year": "..."}} ],
          "semantic_entities": {json.dumps(entities)},
          "warnings": [ "..." ]
        }}
        ```
        """

        try:
            response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
            raw_text = response.text
            
            # --- PARSE JSON INTO DICTIONARY ---
            json_str = ""
            # Correct regex with single backslashes
            json_block = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
            
            if json_block:
                json_str = json_block.group(1).strip()
            else:
                # Fallback in case the AI omits the markdown wrapper
                json_str = raw_text.strip()
                
            # Parse the string into a Python dict so the Orchestrator can access keys
            parsed_result = json.loads(json_str)
            parsed_result["status"] = "Research Complete"
            
            # Ensure warnings is always a list to prevent downstream iteration errors
            if "warnings" not in parsed_result:
                parsed_result["warnings"] = []
                
            return parsed_result
            
        except json.JSONDecodeError as e:
            print(f"🚨 JSON Parsing Error: {e}")
            return {
                "status": "Needs Review: insufficient sources", 
                "warnings": [f"Failed to parse LLM output into JSON: {e}"],
                "raw_brief": raw_text if 'raw_text' in locals() else ""
            }
        except Exception as e:
            print(f"🚨 Researcher Execution Error: {e}")
            return {"status": "Needs Review: insufficient sources", "warnings": [str(e)]}