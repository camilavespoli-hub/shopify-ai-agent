import os
import re
import time
import requests


# ─────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────


def validate_publisher_inputs(content_row, optimizer_result, config):
    """
    Hard-gate validation before publishing.
    Returns {"valid": bool, "errors": [], "warnings": []}
    """
    errors   = []
    warnings = []

    # 1. Optimizer result must be a passing dict
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

    # 2. Content row fields
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

    # 3. Config
    brand = config.get("brand", {})
    if not brand.get("default_author_name"):
        warnings.append(
            "default_author_name missing from Config_Brand — "
            "defaulting to 'Glomend Editorial Team'."
        )
    if not os.getenv("PEXELS_API_KEY"):
        warnings.append(
            "PEXELS_API_KEY not set — featured image will use placeholder."
        )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ─────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────


class PublisherAgent:
    def __init__(self, config=None, optimizer_agent=None):
        """
        Agent 6: Shopify Publisher.

        Tools  : Shopify Admin GraphQL API, Pexels Image API
        Input  : structured dicts from OptimizerAgent + Content_Plan row
        Output : structured dict with status, shopify_id, admin_url

        Key behaviours:
        - Credentials validated in __init__, not at module import time
        - JSON-LD schema prepended to HTML before publish
        - SEO title + description sent via metafields (global namespace)
          — "seo" is not a valid ArticleCreateInput field
        - image.url used (not image.src — invalid field name in Shopify API)
        - summary and tags fields populated from content_row
        - ChromaDB memory updated via optimizer_agent.add_to_memory()
          after every successful publish
        """
        print("🔌 Initializing Shopify Publisher Agent...")
        # ✅ FIX: consistent default with Reviewer and Optimizer
        self.config          = config or {"brand": {}, "system": {}}
        self.optimizer_agent = optimizer_agent

        # ── Credentials — same vars as .env ──────────────────────────
        self.shop          = os.getenv("SHOPIFY_SHOP")
        self.client_id     = os.getenv("SHOPIFY_CLIENT_ID")
        self.client_secret = os.getenv("SHOPIFY_CLIENT_SECRET")
        self.pexels_key    = os.getenv("PEXELS_API_KEY")

        if not self.shop or not self.client_id or not self.client_secret:
            raise RuntimeError(
                "PublisherAgent requires SHOPIFY_SHOP, SHOPIFY_CLIENT_ID, "
                "and SHOPIFY_CLIENT_SECRET environment variables."
            )

        # ── Token cache ──────────────────────────────────────────────
        self._token            = None
        self._token_expires_at = 0.0


    # ── AUTH: Get / Refresh Token ────────────────────────────────────────
    def _get_token(self):
        """Returns a valid Shopify access token, refreshing if expired."""
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


    # ── CORE: GraphQL Dispatcher ─────────────────────────────────────────
    def _graphql(self, query, variables=None):
        """Sends a GraphQL request to the Shopify Admin API."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(
            # ✅ FIX: API version 2024-04 — consistent with Optimizer
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
        """
        Finds the Shopify Blog ID matching section_name.
        Logs a clear warning before falling back to the first blog —
        never fails silently.
        """
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

        # No direct match — log available blogs
        available = [f"'{e['node']['title']}' ({e['node']['handle']})" for e in edges]
        print(f"   ⚠️ No blog matching '{section_name}' found.")
        print(f"   Available blogs: {', '.join(available)}")

        # Try default_section from Config_Brand before falling back to first blog
        default_section = self.config.get("brand", {}).get("default_section", "")
        if default_section:
            default_handle = default_section.lower().replace(" ", "-")
            for edge in edges:
                node = edge["node"]
                if (node["title"].lower()  == default_section.lower() or
                    node["handle"].lower() == default_handle):
                    print(f"   📂 Using default_section fallback: '{node['title']}'")
                    return node["id"]

        # Last resort — first blog in Shopify
        print(f"   ⚠️ Using last-resort fallback: '{edges[0]['node']['title']}'")
        return edges[0]["node"]["id"]



    # ── TOOL 2: Featured Image ───────────────────────────────────────────
    def _get_featured_image(self, keyword):
        """
        Fetches a landscape stock image from Pexels.
        Returns (url, alt_text) — falls back to placeholder gracefully.
        """
        fallback_url = (
            "https://via.placeholder.com/800x600.png?text=Glomend+Wellness"
        )
        fallback_alt = "Glomend women's wellness"

        if not self.pexels_key:
            print("   ⚠️ No PEXELS_API_KEY — using placeholder image.")
            return fallback_url, fallback_alt

        search_term = keyword or "women wellness perimenopause"
        try:
            response = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": search_term, "per_page": 1,
                        "orientation": "landscape"},
                headers={"Authorization": self.pexels_key},
                timeout=10
            )
            if response.status_code == 200:
                photos = response.json().get("photos", [])
                if photos:
                    url = photos[0]["src"]["large"]
                    alt = photos[0].get("alt", search_term)
                    print(f"   🖼️ Image found: {url[:60]}...")
                    return url, alt
        except Exception as e:
            print(f"   ⚠️ Pexels image search failed: {e}")

        return fallback_url, fallback_alt


    # ── TOOL 3: Schema + HTML Assembly ──────────────────────────────────
    def _assemble_body(self, html_content, schema_block):
        """
        Prepends JSON-LD schema <script> blocks to the article HTML.
        Returns html_content unchanged if schema_block is empty.
        """
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
        """
        Full publish pipeline for one Content_Plan row.

        Args:
            content_row      : dict — Title, Section, Keyword, Summary,
                               ScheduledDate from Content_Plan
            optimizer_result : dict — html, meta_title, meta_description,
                               url_slug, schema, word_count, status
                               from OptimizerAgent

        Returns:
            dict — {
                status        : "Draft Created" | "BLOCKED" | "FAILED",
                shopify_id    : str  — raw Shopify article GID,
                admin_url     : str  — direct Admin link for editorial review,
                article_handle: str  — storefront URL handle,
                warnings      : [str],
                errors        : [str]
            }
        """
        if config is None:
            config = self.config

        title = content_row.get("Title", "untitled")
        print(f"\n🚀 Publisher running for: {title}")

        # ── Step 1: Validate inputs ──────────
        validation = validate_publisher_inputs(
            content_row, optimizer_result, config
        )
        if not validation["valid"]:
            print("🚨 Publisher BLOCKED — validation failed:")
            for err in validation["errors"]:
                print(f"   ❌ {err}")
            return {
                "status":         "BLOCKED",
                "shopify_id":     "",
                "admin_url":      "",
                "article_handle": "",
                "warnings":       validation["warnings"],
                "errors":         validation["errors"]
            }

        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"   ⚠ {w}")

        # ── Step 2: Gather publish data ──────
        brand        = config.get("brand", {})
        default_section = config.get("brand", {}).get("default_section", "")
        section_name    = content_row.get("Section", "") or default_section
        keyword      = content_row.get("Keyword",  title)
        summary      = content_row.get("Summary",  "")
        author_name  = brand.get("default_author_name", "Glomend Editorial Team")

        # Visibility — controlled from Content_Plan sheet (col Visibility)
        visibility = content_row.get("Visibility") or \
            config.get("brand", {}).get("default_visibility", "Public")
        is_public  = visibility.strip().lower() == "public"

        html_content = optimizer_result.get("html",             "")
        schema_block = optimizer_result.get("schema",           "")
        meta_title   = optimizer_result.get("meta_title",       title)
        meta_desc    = optimizer_result.get("meta_description", "")
        url_slug     = optimizer_result.get("url_slug",         "")

        if url_slug:
            url_slug = re.sub(
                r'[^a-z0-9-]',
                '',
                url_slug.lower().replace(" ", "-")
            )

        # ── Step 3: Assemble final HTML with schema ──
        final_body = self._assemble_body(html_content, schema_block)

        # ── Step 4: Get featured image ───────
        image_url, image_alt = self._get_featured_image(keyword)

        # ── Step 5: Find Blog ID ─────────────
        try:
            blog_id = self._get_blog_id_by_name(section_name)
        except Exception as e:
            print(f"   🚨 Could not find Blog ID: {e}")
            return {
                "status":         "FAILED",
                "shopify_id":     "",
                "admin_url":      "",
                "article_handle": "",
                "warnings":       validation["warnings"],
                "errors":         [f"Blog lookup failed: {str(e)}"]
            }

        # ── Step 6: Build GraphQL mutation ───
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
                "namespace": "global",
                "key":       "title_tag",
                "value":     meta_title[:60],
                "type":      "single_line_text_field"
            })
        if meta_desc:
            metafields.append({
                "namespace": "global",
                "key":       "description_tag",
                "value":     meta_desc[:155],
                "type":      "single_line_text_field"
            })

        tags = [
            t.strip()
            for t in keyword.split(",")
            if t.strip()
        ][:5]

        variables = {
            "article": {
                "title":       title,
                "body":        final_body,
                "blogId":      blog_id,
                "author":      {"name": author_name},
                "isPublished": is_public,
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

        # ── Step 7: Publish to Shopify ───────
        print(f"   📤 Pushing '{title}' to Shopify...")
        try:
            result = self._graphql(mutation, variables)
        except Exception as e:
            print(f"   🚨 GraphQL publish failed: {e}")
            return {
                "status":         "FAILED",
                "shopify_id":     "",
                "admin_url":      "",
                "article_handle": "",
                "warnings":       validation["warnings"],
                "errors":         [f"GraphQL publish failed: {str(e)}"]
            }

        # ── Step 8: Handle Shopify response ──
        user_errors = result.get("articleCreate", {}).get("userErrors", [])
        if user_errors:
            msg = user_errors[0]["message"]
            print(f"   ❌ Shopify userError: {msg}")
            return {
                "status":         "FAILED",
                "shopify_id":     "",
                "admin_url":      "",
                "article_handle": "",
                "warnings":       validation["warnings"],
                "errors":         [f"Shopify userError: {msg}"]
            }

        article   = result["articleCreate"]["article"]
        raw_id    = article["id"]
        handle    = article.get("handle", "")
        clean_id  = raw_id.split("/")[-1]
        admin_url = (
            f"https://{self.shop}.myshopify.com"
            f"/admin/articles/{clean_id}"
        )
        print(f"   ✅ Draft created: {admin_url}")

        # ── Step 9: Update ChromaDB memory ───
        if self.optimizer_agent and summary:
            blog_handle    = section_name.lower().replace(" ", "-")
            storefront_url = f"/blogs/{blog_handle}/{handle}"
            self.optimizer_agent.add_to_memory(title, summary, storefront_url)
        elif not summary:
            validation["warnings"].append(
                "NEEDS REVIEW: ChromaDB memory NOT updated — "
                "Summary was missing from content_row."
            )

        # ── Step 10: Build final warnings ────
        final_warnings = list(validation["warnings"])
        if optimizer_result.get("status") == "Needs Review":
            final_warnings.append(
                "NEEDS REVIEW: Optimizer flagged this article before publish. "
                "Check ReviewerNotes in Google Sheet."
            )

        # ── Step 11: Report ──────────────────
        print(f"✅ Publish complete.")
        print(f"   Status     : Draft Created")
        print(f"   Admin URL  : {admin_url}")
        print(f"   Handle     : {handle}")
        if final_warnings:
            for w in final_warnings:
                print(f"   ⚠ {w}")

        return {
            "status":         "Draft Created",
            "shopify_id":     raw_id,
            "admin_url":      admin_url,
            "article_handle": handle,
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
            "Title":   "Why Does Perimenopause Cause Night Sweats?",
            "Section": "News",
            "Keyword": "perimenopause night sweats",
            "Summary": "Explains how estrogen fluctuations affect the hypothalamus."
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
