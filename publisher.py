import os
import re
import time
import requests


# ─────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────

def validate_publisher_inputs(content_row, optimizer_result, config):
    errors   = []
    warnings = []

    if not optimizer_result or not isinstance(optimizer_result, dict):
        errors.append(
            "optimizer_result must be a dict from OptimizerAgent. "
            f"Got: {type(optimizer_result).__name__}"
        )
    else:
        status = optimizer_result.get("status", "")
        if status == "BLOCKED":
            errors.append(
                f"Cannot publish: Optimizer was BLOCKED. "
                f"Errors: {optimizer_result.get('errors', [])}"
            )
        if not optimizer_result.get("html"):
            errors.append(
                "optimizer_result is missing 'html'. "
                "Cannot publish an empty article."
            )
        if not optimizer_result.get("meta_title"):
            warnings.append(
                "NEEDS REVIEW: meta_title missing — "
                "Shopify SEO title will fall back to article title."
            )
        if not optimizer_result.get("meta_description"):
            warnings.append(
                "NEEDS REVIEW: meta_description missing — "
                "Shopify SEO description will be blank."
            )
        if not optimizer_result.get("url_slug"):
            warnings.append(
                "NEEDS REVIEW: url_slug missing — "
                "Shopify will auto-generate a handle."
            )

    if not content_row.get("Title"):
        errors.append("Title is required in content_row.")
    if not content_row.get("Section"):
        warnings.append(
            "Section missing from content_row — "
            "Publisher will use default blog."
        )
    if not content_row.get("Keyword"):
        warnings.append(
            "Keyword missing — featured image search will use title words."
        )
    if not content_row.get("Summary"):
        warnings.append(
            "Summary missing — ChromaDB memory will not be updated after publish."
        )

    brand = config.get("brand", {})
    if not brand.get("default_author_name"):
        warnings.append(
            "default_author_name missing from Config_Brand — "
            "defaulting to 'Glomend Editorial Team'."
        )

    # ✅ Updated: warn if BOTH image sources are missing
    has_pexels   = bool(os.getenv("PEXELS_API_KEY"))
    has_unsplash = bool(os.getenv("UNSPLASH_ACCESS_KEY"))
    if not has_pexels and not has_unsplash:
        warnings.append(
            "Neither PEXELS_API_KEY nor UNSPLASH_ACCESS_KEY is set — "
            "featured image will use placeholder."
        )
    elif not has_pexels:
        warnings.append(
            "PEXELS_API_KEY not set — image source will be Unsplash only."
        )
    elif not has_unsplash:
        warnings.append(
            "UNSPLASH_ACCESS_KEY not set — no image fallback available if Pexels fails."
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ─────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────

class PublisherAgent:

    def __init__(self, config=None, optimizer_agent=None):
        print("🔌 Initializing Shopify Publisher Agent...")
        self.config          = config or {"brand": {}, "system": {}}
        self.optimizer_agent = optimizer_agent

        self.shop          = os.getenv("SHOPIFY_SHOP")
        self.client_id     = os.getenv("SHOPIFY_CLIENT_ID")
        self.client_secret = os.getenv("SHOPIFY_CLIENT_SECRET")
        self.pexels_key    = os.getenv("PEXELS_API_KEY")
        self.unsplash_key  = os.getenv("UNSPLASH_ACCESS_KEY")   # ✅ NEW

        if not self.shop or not self.client_id or not self.client_secret:
            raise RuntimeError(
                "PublisherAgent requires SHOPIFY_SHOP, SHOPIFY_CLIENT_ID, "
                "and SHOPIFY_CLIENT_SECRET environment variables."
            )

        self._token            = None
        self._token_expires_at = 0.0


    # ── AUTH ─────────────────────────────────────────────────────────────
    def _get_token(self):
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        response = requests.post(
            f"https://{self.shop}.myshopify.com/admin/oauth/access_token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type":    "client_credentials",
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        response.raise_for_status()
        data                   = response.json()
        self._token            = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 3600)
        return self._token


    # ── GRAPHQL ──────────────────────────────────────────────────────────
    def _graphql(self, query, variables=None):
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(
            f"https://{self.shop}.myshopify.com/admin/api/2024-04/graphql.json",
            headers={
                "Content-Type":           "application/json",
                "X-Shopify-Access-Token": self._get_token(),
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("errors"):
            print(f"   🚨 Shopify GraphQL error: {data['errors']}")
            raise RuntimeError(data["errors"])

        return data["data"]


    # ── TOOL 1: Find Blog ID ─────────────────────────────────────────────
    def _get_blog_id_by_name(self, section_name):
        query = "{ blogs(first: 10) { edges { node { id title handle } } } }"
        data  = self._graphql(query)
        edges = data.get("blogs", {}).get("edges", [])

        if not edges:
            raise ValueError(
                "No blogs found in Shopify. "
                "Create at least one blog before publishing."
            )

        section_handle = section_name.lower().replace(" ", "-")
        for edge in edges:
            node = edge["node"]
            if (node["title"].lower()  == section_name.lower() or
                node["handle"].lower() == section_handle):
                print(f"   📂 Blog matched: '{node['title']}' (handle: {node['handle']})")
                return node["id"]

        available = [f"'{e['node']['title']}' ({e['node']['handle']})" for e in edges]
        print(f"   ⚠️ No blog matching '{section_name}' found.")
        print(f"   Available blogs: {', '.join(available)}")

        default_section = self.config.get("brand", {}).get("default_section", "")
        if default_section:
            default_handle = default_section.lower().replace(" ", "-")
            for edge in edges:
                node = edge["node"]
                if (node["title"].lower()  == default_section.lower() or
                    node["handle"].lower() == default_handle):
                    print(f"   📂 Using default_section fallback: '{node['title']}'")
                    return node["id"]

        print(f"   ⚠️ Using last-resort fallback: '{edges[0]['node']['title']}'")
        return edges[0]["node"]["id"]


    # ── TOOL 2a: Pexels Search ───────────────────────────────────────────
    def _search_pexels(self, search_term: str) -> tuple:
        """
        Returns (url, alt_text) from Pexels or (None, None) if not found.
        """
        try:
            response = requests.get(
                "https://api.pexels.com/v1/search",
                params={
                    "query":       search_term,
                    "per_page":    1,
                    "orientation": "landscape"
                },
                headers={"Authorization": self.pexels_key},
                timeout=10
            )
            if response.status_code == 200:
                photos = response.json().get("photos", [])
                if photos:
                    url = photos[0]["src"]["large"]
                    alt = photos[0].get("alt", search_term)
                    print(f"   🖼️  [Pexels] Image found: {url[:60]}...")
                    return url, alt
            print(f"   ⚠️  Pexels: no results for '{search_term}'.")
        except Exception as e:
            print(f"   ⚠️  Pexels error: {e}")
        return None, None


    # ── TOOL 2b: Unsplash Search ─────────────────────────────────────────
    def _search_unsplash(self, search_term: str) -> tuple:
        """
        Returns (url, alt_text) from Unsplash or (None, None) if not found.
        Rate limit: 50 req/hour on free tier.
        """
        try:
            response = requests.get(
                "https://api.unsplash.com/search/photos",
                params={
                    "query":          search_term,
                    "per_page":       1,
                    "orientation":    "landscape",
                    "content_filter": "high"   # safe content only
                },
                headers={"Authorization": f"Client-ID {self.unsplash_key}"},
                timeout=10
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    photo = results[0]
                    url   = photo["urls"]["regular"]
                    alt   = photo.get("alt_description") or search_term
                    print(f"   🖼️  [Unsplash] Image found: {url[:60]}...")
                    return url, alt
            print(f"   ⚠️  Unsplash: no results for '{search_term}'.")
        except Exception as e:
            print(f"   ⚠️  Unsplash error: {e}")
        return None, None


    # ── TOOL 2: Featured Image (Pexels → Unsplash → placeholder) ─────────
    def _get_featured_image(self, keyword: str) -> tuple:
        """
        Tries Pexels first, falls back to Unsplash, then placeholder.
        Returns (url, alt_text) — never raises.
        """
        brand        = self.config.get("brand", {})
        brand_name   = brand.get("brand_name", "Glomend")
        fallback_url = f"https://via.placeholder.com/800x600.png?text={brand_name}+Wellness"
        fallback_alt = f"{brand_name} women's wellness"

        search_term = keyword or "women wellness perimenopause"

        # ── Attempt 1: Pexels ─────────────────
        if self.pexels_key:
            url, alt = self._search_pexels(search_term)
            if url:
                return url, alt
            print("   🔄 Falling back to Unsplash...")
        else:
            print("   ⚠️  No PEXELS_API_KEY — skipping Pexels.")

        # ── Attempt 2: Unsplash ───────────────
        if self.unsplash_key:
            url, alt = self._search_unsplash(search_term)
            if url:
                return url, alt
            print("   ⚠️  Unsplash also returned no results.")
        else:
            print("   ⚠️  No UNSPLASH_ACCESS_KEY — skipping Unsplash.")

        # ── Attempt 3: Placeholder ────────────
        print(f"   🖼️  Using placeholder image.")
        return fallback_url, fallback_alt


    # ── TOOL 3: Schema + HTML Assembly ──────────────────────────────────
    def _assemble_body(self, html_content, schema_block):
        if not schema_block or not schema_block.strip():
            return html_content

        schema_text = schema_block.strip()
        if not schema_text.startswith("<script"):
            schema_text = (
                f'<script type="application/ld+json">\n'
                f'{schema_text}\n'
                f'</script>'
            )

        return f"{schema_text}\n\n{html_content}"


    # ── MAIN: publish_post ───────────────────────────────────────────────
    def publish_post(self, content_row, optimizer_result, config=None):
        if config is None:
            config = self.config

        title = content_row.get("Title", "untitled")
        print(f"\n🚀 Publisher running for: {title}")

        # Step 1: Validate
        validation = validate_publisher_inputs(content_row, optimizer_result, config)
        if not validation["valid"]:
            print("🚨 Publisher BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"   ❌ {err}")
            return {
                "status": "BLOCKED", "shopify_id": "", "admin_url": "",
                "article_handle": "", "is_hidden": False, "hidden_reason": "",
                "warnings": validation["warnings"], "errors": validation["errors"]
            }

        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"   ⚠ {w}")

        # Step 2: Gather data
        brand           = config.get("brand", {})
        default_section = brand.get("default_section", "")
        section_name    = content_row.get("Section", "") or default_section
        keyword         = content_row.get("Keyword",  title)
        summary         = content_row.get("Summary",  "")
        author_name     = brand.get("default_author_name", "Glomend Editorial Team")

        visibility_raw = str(content_row.get("Visibility", "")).strip().lower()
        is_hidden      = visibility_raw == "hidden"
        hidden_reason  = (
            "Max retries reached — content has unresolved reviewer flags."
            if is_hidden else ""
        )

        html_content = optimizer_result.get("html",             "")
        schema_block = optimizer_result.get("schema",           "")
        meta_title   = optimizer_result.get("meta_title",       title)
        meta_desc    = optimizer_result.get("meta_description", "")
        url_slug     = optimizer_result.get("url_slug",         "")

        if url_slug:
            url_slug = re.sub(r'[^a-z0-9-]', '', url_slug.lower().replace(" ", "-"))

        # Step 3: Assemble HTML
        final_body = self._assemble_body(html_content, schema_block)

        # Step 4: Get image (Pexels → Unsplash → placeholder)
        image_url, image_alt = self._get_featured_image(keyword)

        # Step 5: Find Blog ID
        try:
            blog_id = self._get_blog_id_by_name(section_name)
        except Exception as e:
            print(f"   🚨 Could not find Blog ID: {e}")
            return {
                "status": "FAILED", "shopify_id": "", "admin_url": "",
                "article_handle": "", "is_hidden": False, "hidden_reason": "",
                "warnings": validation["warnings"],
                "errors": [f"Blog lookup failed: {str(e)}"]
            }

        # Step 6: Build mutation
        mutation = """
        mutation articleCreate($article: ArticleCreateInput!) {
            articleCreate(article: $article) {
                article {
                    id
                    title
                    handle
                }
                userErrors {
                    field
                    message
                }
            }
        }
        """

        metafields = []
        if meta_title:
            metafields.append({
                "namespace": "global", "key": "title_tag",
                "value": meta_title[:60], "type": "single_line_text_field"
            })
        if meta_desc:
            metafields.append({
                "namespace": "global", "key": "description_tag",
                "value": meta_desc[:155], "type": "single_line_text_field"
            })

        tags = [t.strip() for t in keyword.split(",") if t.strip()][:5]
        if is_hidden:
            tags.append("needs-review")

        variables = {
            "article": {
                "title":       title,
                "body":        final_body,
                "blogId":      blog_id,
                "author":      {"name": author_name},
                "isPublished": True,
                "summary":     summary[:500] if summary else "",
                "tags":        tags,
                **({"handle": url_slug} if url_slug else {}),
                "image": {
                    "url":     image_url,
                    "altText": image_alt
                },
                **({"metafields": metafields} if metafields else {})
            }
        }

        # Step 7: Publish
        live_label = "Live/Hidden (needs-review)" if is_hidden else "Live/Public"
        print(f"   📤 Pushing '{title}' to Shopify as {live_label}...")
        try:
            result = self._graphql(mutation, variables)
        except Exception as e:
            print(f"   🚨 GraphQL publish failed: {e}")
            return {
                "status": "FAILED", "shopify_id": "", "admin_url": "",
                "article_handle": "", "is_hidden": False, "hidden_reason": "",
                "warnings": validation["warnings"],
                "errors": [f"GraphQL publish failed: {str(e)}"]
            }

        # Step 8: Handle response
        user_errors = result.get("articleCreate", {}).get("userErrors", [])
        if user_errors:
            msg = user_errors[0]["message"]
            print(f"   ❌ Shopify userError: {msg}")
            return {
                "status": "FAILED", "shopify_id": "", "admin_url": "",
                "article_handle": "", "is_hidden": False, "hidden_reason": "",
                "warnings": validation["warnings"],
                "errors": [f"Shopify userError: {msg}"]
            }

        article   = result["articleCreate"]["article"]
        raw_id    = article["id"]
        handle    = article.get("handle", "")
        clean_id  = raw_id.split("/")[-1]
        admin_url = f"https://{self.shop}.myshopify.com/admin/articles/{clean_id}"

        # Step 9: ChromaDB
        if self.optimizer_agent and summary and not is_hidden:
            blog_handle    = section_name.lower().replace(" ", "-")
            storefront_url = f"/blogs/{blog_handle}/{handle}"
            self.optimizer_agent.add_to_memory(title, summary, storefront_url)
        elif is_hidden:
            validation["warnings"].append(
                "ChromaDB memory NOT updated — article published as Hidden. "
                "Update manually after editorial review."
            )
        elif not summary:
            validation["warnings"].append(
                "ChromaDB memory NOT updated — Summary was missing."
            )

        # Step 10: Final warnings
        final_warnings = list(validation["warnings"])
        if optimizer_result.get("status") == "Needs Review":
            final_warnings.append(
                "NEEDS REVIEW: Optimizer flagged this article before publish. "
                "Check ReviewerNotes in Google Sheet."
            )

        # Step 11: Report
        print(f"✅ Published: {live_label}")
        print(f"   Admin URL  : {admin_url}")
        print(f"   Handle     : {handle}")
        if is_hidden:
            print(f"   ⚠️  Reason   : {hidden_reason}")

        return {
            "status":         "Live",
            "shopify_id":     raw_id,
            "admin_url":      admin_url,
            "article_handle": handle,
            "is_hidden":      is_hidden,
            "hidden_reason":  hidden_reason,
            "warnings":       final_warnings,
            "errors":         []
        }


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = PublisherAgent()

    result = agent.publish_post(
        content_row = {
            "Title":      "Why Does Perimenopause Cause Night Sweats?",
            "Section":    "Sleep",
            "Keyword":    "perimenopause night sweats",
            "Summary":    "Explains how estrogen fluctuations affect the hypothalamus.",
            "Visibility": ""
        },
        optimizer_result = {
            "status":           "Ready for Approval",
            "html":             "<p>Test article body.</p>",
            "schema":           "",
            "meta_title":       "Why Does Perimenopause Cause Night Sweats? | Glomend",
            "meta_description": "Learn how estrogen changes trigger night sweats.",
            "url_slug":         "perimenopause-night-sweats-causes",
            "word_count":       350
        }
    )
    print(result)
