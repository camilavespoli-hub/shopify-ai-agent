import os
import io
import re
import json
import time
import textwrap
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types as genai_types


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE AGENT — Agent 5.5 of the pipeline (runs between Optimizer and Publisher)
#
# Responsibilities:
#   1. Generate a featured (hero) image with fal.ai flux-2-pro
#      (art direction rules ported from glomend-imagegen/IMAGE-RULES.md)
#   2. Generate 1 inline lifestyle image for the article body
#   3. AI QA REVIEWER: every generated image is reviewed by Gemini vision
#      against the brand checklist — rejected images are regenerated
#      (max attempts capped to control cost)
#   4. Build 1 branded infographic from the article's key facts
#      (deterministic Pillow template — zero typo risk, always on-brand)
#   5. Upload all images to Shopify Files (staged upload → fileCreate → CDN)
#   6. Insert inline <img> tags with keyword-optimized alt text into the HTML
#
# Design rules:
#   - NEVER blocks publishing: any failure degrades gracefully — the article
#     publishes with whatever images succeeded (Publisher falls back to
#     stock photos for the featured image if the hero fails).
#   - Cost per article ≈ 2 flux-2-pro generations (~$0.03–0.05 each)
#     + retries when QA rejects + 2–3 cheap Gemini flash vision calls.
#   - Blog images NEVER include the product bottle (label fidelity requires
#     the reference-image pipeline in glomend-imagegen — out of scope here).
# ══════════════════════════════════════════════════════════════════════════════


FAL_MODEL_DEFAULT    = "fal-ai/flux-2-pro"
REVIEW_MODEL_DEFAULT = "gemini-2.5-flash"

# ── Art direction (ported from glomend-imagegen WOMAN_RULES) ──────────────────
ART_DIRECTION = """Ultra-realistic editorial photograph, NOT stock photography.
Woman aged 45-55 with genuine skin texture: visible pores, fine lines, natural
asymmetry. Absolutely no plastic, airbrushed, or CGI-looking skin. Natural
visible makeup (neither bare-faced nor glam). Short well-kept nails, natural
or light pastel color — never long nails. High-agency context (home office,
kitchen, desk, bedroom) — never a spa. Shot on 85mm lens, soft natural window
light, subtle film grain. No third-party brand logos anywhere. NO text,
lettering, or typography anywhere in the image."""

QA_CHECKLIST = """You are a strict brand QA reviewer for Glomend, a wellness
brand for women in perimenopause. Review this AI-generated editorial image.

REJECT the image if ANY of these fail (reject when in doubt):
1. Anatomy errors: wrong hands, fingers, teeth, eyes, limbs.
2. Plastic/airbrushed/CGI skin — real skin must show pores and fine lines.
3. Any AI-generated text, lettering, or garbled typography in the image.
4. Stock-photo or glamour look (fake smiles, posed perfection, spa setting).
5. Long nails, or loud nail polish.
6. Third-party brand logos visible.
7. Image is irrelevant to the article topic given below.
8. Harsh studio lighting or collage/split-screen look — must feel like one
   naturally lit editorial photograph.

Respond ONLY with JSON: {"approved": true/false, "reason": "one short sentence"}"""


class ImageAgent:

    def __init__(self, config=None):
        print("🎨 Initializing Image Agent...")
        self.config = config or {"brand": {}, "system": {}}

        # ── fal.ai (generation) ────────────────────────────────────────────
        self.fal_key   = os.getenv("FAL_KEY")
        self.fal_model = os.getenv("FAL_IMAGE_MODEL", FAL_MODEL_DEFAULT)
        if not self.fal_key:
            print("   ⚠️  FAL_KEY missing — image generation disabled "
                  "(infographic still works).")

        # ── Gemini (QA review + fact extraction) ───────────────────────────
        api_key = os.getenv("GEMINI_API_KEY")
        self.client       = genai.Client(api_key=api_key) if api_key else None
        self.review_model = os.getenv("GEMINI_REVIEW_MODEL", REVIEW_MODEL_DEFAULT)

        # ── Shopify (upload) — same client-credentials flow as Publisher ───
        self.shop          = os.getenv("SHOPIFY_SHOP")
        self.client_id     = os.getenv("SHOPIFY_CLIENT_ID")
        self.client_secret = os.getenv("SHOPIFY_CLIENT_SECRET")
        self._token            = None
        self._token_expires_at = 0.0

        self.max_gen_attempts = int(os.getenv("IMAGE_QA_MAX_ATTEMPTS", "3"))

        # ── Brand palette (same keys the Optimizer uses) ───────────────────
        brand = self.config.get("brand", {})
        self.palette = {
            "teal":  brand.get("h2_color",             "#18655C"),
            "plum":  brand.get("faq_title_color",      "#4B2E4A"),
            "ink":   brand.get("body_text_color",      "#121212"),
            "muted": brand.get("fda_disclaimer_color", "#575654"),
            "cream": brand.get("infographic_bg_color", "#FAF6F0"),
        }
        self.brand_name = brand.get("brand_name", "Glomend")


    # ══════════════════════════════════════════════════════════════════════
    # GENERATION — fal.ai flux-2-pro via REST (no extra dependency)
    # ══════════════════════════════════════════════════════════════════════

    def _fal_generate(self, prompt, image_size="landscape_16_9"):
        """
        Calls fal.ai synchronously. Returns (bytes, fal_url) or (None, "").
        The fal.media URL is kept as a fallback featured-image source when
        Shopify Files upload is unavailable (missing write_files scope) —
        articleCreate ingests external URLs into Shopify's own CDN.
        flux-2-pro ≈ $0.03–0.05 per image.
        """
        if not self.fal_key:
            return None, ""
        try:
            response = requests.post(
                f"https://fal.run/{self.fal_model}",
                headers={
                    "Authorization": f"Key {self.fal_key}",
                    "Content-Type":  "application/json",
                },
                json={"prompt": prompt, "image_size": image_size},
                timeout=180,
            )
            response.raise_for_status()
            images = response.json().get("images", [])
            if not images:
                print("   ⚠️  fal.ai returned no images.")
                return None, ""
            img_url = images[0]["url"]
            img_bytes = requests.get(img_url, timeout=60).content
            return img_bytes, img_url
        except Exception as e:
            print(f"   ⚠️  fal.ai generation failed: {e}")
            return None, ""


    # ══════════════════════════════════════════════════════════════════════
    # QA REVIEWER — Gemini vision approves/rejects every generated image
    # ══════════════════════════════════════════════════════════════════════

    def review_image(self, image_bytes, article_title, image_purpose):
        """
        AI QA gate. Returns {"approved": bool, "reason": str}.
        If the reviewer itself fails, the image is approved with a warning —
        a missing reviewer must never block the pipeline.
        """
        if not self.client:
            return {"approved": True, "reason": "Reviewer unavailable — auto-approved."}
        try:
            raw = self.client.models.generate_content(
                model=self.review_model,
                contents=[
                    genai_types.Part.from_bytes(
                        data=image_bytes, mime_type="image/jpeg"
                    ),
                    f"{QA_CHECKLIST}\n\nARTICLE TOPIC: {article_title}\n"
                    f"IMAGE PURPOSE: {image_purpose}",
                ],
            ).text.strip()
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
            verdict = json.loads(raw)
            return {
                "approved": bool(verdict.get("approved", False)),
                "reason":   str(verdict.get("reason", "no reason given")),
            }
        except Exception as e:
            print(f"   ⚠️  QA reviewer error — approving by default: {e}")
            return {"approved": True, "reason": f"Reviewer error: {e}"}


    def generate_with_qa(self, prompt, article_title, image_purpose,
                          image_size="landscape_16_9"):
        """
        Generate → QA review → regenerate loop (max self.max_gen_attempts).
        On rejection, the QA reason is appended to the prompt as negative
        feedback for the next attempt. Returns (bytes, fal_url) or (None, "").
        """
        feedback = ""
        for attempt in range(1, self.max_gen_attempts + 1):
            print(f"   🎨 Generating {image_purpose} (attempt {attempt}/{self.max_gen_attempts})...")
            img, fal_url = self._fal_generate(prompt + feedback, image_size=image_size)
            if not img:
                return None, ""

            verdict = self.review_image(img, article_title, image_purpose)
            if verdict["approved"]:
                print(f"   ✅ QA approved: {verdict['reason']}")
                return img, fal_url

            print(f"   ❌ QA rejected: {verdict['reason']}")
            feedback = (
                f"\n\nIMPORTANT — the previous attempt was rejected for this "
                f"reason, fix it: {verdict['reason']}"
            )
        print(f"   ⚠️  All {self.max_gen_attempts} attempts rejected by QA.")
        return None, ""


    # ══════════════════════════════════════════════════════════════════════
    # INFOGRAPHIC — deterministic Pillow template (no AI text = no typos)
    # ══════════════════════════════════════════════════════════════════════

    def _font(self, size, bold=False):
        """Loads a system TTF font, falling back to Pillow's scalable default."""
        candidates = (
            [  # macOS
                "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold
                else "/System/Library/Fonts/Supplemental/Georgia.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
                else "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
            + [  # Linux (Railway/nixpacks)
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)


    def extract_infographic_data(self, html, title, keyword):
        """
        Uses Gemini flash to pull a short headline + 4 key takeaways
        (with a leading stat where available) from the finished article.
        Returns {"headline": str, "facts": [str, ...]} or None.
        """
        if not self.client:
            return None
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)[:8000]
        prompt = f"""Extract infographic content from this article.

ARTICLE TITLE: {title}
PRIMARY KEYWORD: {keyword}
ARTICLE TEXT: {text}

Return ONLY JSON:
{{
  "headline": "[short punchy version of the article's core answer, max 60 chars]",
  "facts": [
    "[key takeaway 1 — max 110 chars, lead with a number/stat when the article has one]",
    "[key takeaway 2]",
    "[key takeaway 3]",
    "[key takeaway 4]"
  ]
}}

Rules: only facts actually stated in the article. No disease claims
(no treat/cure/prevent/diagnose). Plain language a reader scans in 3 seconds."""
        try:
            raw = self.client.models.generate_content(
                model=self.review_model,
                contents=prompt,
            ).text.strip()
            raw  = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            if data.get("headline") and data.get("facts"):
                return data
        except Exception as e:
            print(f"   ⚠️  Infographic data extraction failed: {e}")
        return None


    def render_infographic(self, data):
        """
        Renders a 1080×1350 branded infographic PNG with Pillow.
        Layout: teal header band → headline → numbered fact cards → footer.
        Deterministic: brand colors, real fonts, zero AI-typo risk.
        """
        W, H = 1080, 1350
        img  = Image.new("RGB", (W, H), self.palette["cream"])
        d    = ImageDraw.Draw(img)

        f_brand    = self._font(34, bold=True)
        f_headline = self._font(56, bold=True)
        f_num      = self._font(44, bold=True)
        f_fact     = self._font(36)
        f_footer   = self._font(30)

        # ── Header band ────────────────────────────────────────────────────
        d.rectangle([0, 0, W, 120], fill=self.palette["teal"])
        d.text((60, 42), self.brand_name.upper(), font=f_brand, fill="#FFFFFF")

        # ── Headline ───────────────────────────────────────────────────────
        y = 180
        for line in textwrap.wrap(data["headline"], width=30):
            d.text((60, y), line, font=f_headline, fill=self.palette["plum"])
            y += 74
        y += 30
        d.rectangle([60, y, 240, y + 6], fill=self.palette["teal"])
        y += 60

        # ── Fact cards ─────────────────────────────────────────────────────
        facts = data["facts"][:4]
        card_h = max(160, (H - y - 140) // len(facts) - 30)
        for i, fact in enumerate(facts, start=1):
            top = y
            d.rounded_rectangle(
                [60, top, W - 60, top + card_h],
                radius=24, fill="#FFFFFF",
                outline=self.palette["teal"], width=2,
            )
            # numbered circle
            cx, cy, r = 140, top + card_h // 2, 44
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=self.palette["teal"])
            num_box = d.textbbox((0, 0), str(i), font=f_num)
            d.text(
                (cx - (num_box[2] - num_box[0]) / 2,
                 cy - (num_box[3] - num_box[1]) / 2 - num_box[1]),
                str(i), font=f_num, fill="#FFFFFF",
            )
            # wrapped fact text, vertically centered
            lines = textwrap.wrap(fact, width=42)[:3]
            line_h = 48
            ty = cy - (len(lines) * line_h) // 2 + 4
            for line in lines:
                d.text((220, ty), line, font=f_fact, fill=self.palette["ink"])
                ty += line_h

            y = top + card_h + 30

        # ── Footer ─────────────────────────────────────────────────────────
        d.rectangle([0, H - 90, W, H], fill=self.palette["plum"])
        footer = f"{self.brand_name.lower()}.com"
        fb = d.textbbox((0, 0), footer, font=f_footer)
        d.text(((W - (fb[2] - fb[0])) / 2, H - 68), footer,
               font=f_footer, fill="#FFFFFF")

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


    # ══════════════════════════════════════════════════════════════════════
    # SHOPIFY UPLOAD — staged upload → fileCreate → poll for CDN URL
    # ══════════════════════════════════════════════════════════════════════

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
            raise RuntimeError(data["errors"])
        return data["data"]


    def upload_image(self, image_bytes, filename, alt_text,
                     mime_type="image/png"):
        """
        Uploads image bytes to Shopify Files. Returns the CDN URL or None.
        Flow: stagedUploadsCreate → POST to staged target → fileCreate → poll.
        NOTE: fal.ai returns JPEG — pass mime_type="image/jpeg" for those.
        """
        if not (self.shop and self.client_id and self.client_secret):
            print("   ⚠️  Shopify credentials missing — cannot upload image.")
            return None
        try:
            # 1. Staged upload target
            staged = self._graphql(
                """
                mutation stagedUploadsCreate($input: [StagedUploadInput!]!) {
                    stagedUploadsCreate(input: $input) {
                        stagedTargets {
                            url resourceUrl
                            parameters { name value }
                        }
                        userErrors { field message }
                    }
                }
                """,
                {"input": [{
                    "resource":   "FILE",
                    "filename":   filename,
                    "mimeType":   mime_type,
                    "httpMethod": "POST",
                }]},
            )["stagedUploadsCreate"]
            if staged.get("userErrors"):
                raise RuntimeError(staged["userErrors"])
            target = staged["stagedTargets"][0]

            # 2. Upload the bytes to the staged target
            form = {p["name"]: p["value"] for p in target["parameters"]}
            up = requests.post(
                target["url"], data=form,
                files={"file": (filename, image_bytes, mime_type)},
                timeout=120,
            )
            up.raise_for_status()

            # 3. Register the file
            created = self._graphql(
                """
                mutation fileCreate($files: [FileCreateInput!]!) {
                    fileCreate(files: $files) {
                        files {
                            id fileStatus
                            ... on MediaImage { image { url } }
                        }
                        userErrors { field message }
                    }
                }
                """,
                {"files": [{
                    "alt":            alt_text,
                    "contentType":    "IMAGE",
                    "originalSource": target["resourceUrl"],
                }]},
            )["fileCreate"]
            if created.get("userErrors"):
                raise RuntimeError(created["userErrors"])
            file_node = created["files"][0]
            file_id   = file_node["id"]

            # 4. Poll until Shopify finishes processing and a CDN URL exists
            for _ in range(15):
                image = (file_node.get("image") or {})
                if file_node.get("fileStatus") == "READY" and image.get("url"):
                    print(f"   ☁️  Uploaded to Shopify CDN: {image['url'][:70]}...")
                    return image["url"]
                time.sleep(2)
                file_node = self._graphql(
                    """
                    query getFile($id: ID!) {
                        node(id: $id) {
                            ... on MediaImage { fileStatus image { url } }
                        }
                    }
                    """,
                    {"id": file_id},
                )["node"] or {}

            print("   ⚠️  Shopify file never reached READY state.")
            return None
        except Exception as e:
            print(f"   ⚠️  Shopify image upload failed: {e}")
            return None


    # ══════════════════════════════════════════════════════════════════════
    # HTML INSERTION
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _img_tag(soup, url, alt):
        tag = soup.new_tag(
            "img", src=url, alt=alt,
            style="max-width:100%;height:auto;border-radius:8px;margin:24px 0;",
        )
        tag["loading"] = "lazy"
        return tag

    def insert_images(self, html, inline_url, inline_alt,
                       infographic_url, infographic_alt):
        """
        Inserts inline <img> tags into the optimized HTML:
          - inline lifestyle image → after the first <h2>'s first paragraph
          - infographic            → right before the FAQ (or Sources) section
        Returns the modified HTML string.
        """
        soup = BeautifulSoup(html, "html.parser")

        if inline_url:
            first_h2 = soup.find("h2")
            anchor = first_h2.find_next("p") if first_h2 else None
            if anchor:
                anchor.insert_after(self._img_tag(soup, inline_url, inline_alt))
            else:
                soup.append(self._img_tag(soup, inline_url, inline_alt))

        if infographic_url:
            target = (soup.find("section", class_="faq")
                      or soup.find("section", class_="sources"))
            tag = self._img_tag(soup, infographic_url, infographic_alt)
            if target:
                target.insert_before(tag)
            else:
                soup.append(tag)

        return str(soup)


    # ══════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════

    def process_article(self, content_row, optimizer_result):
        """
        Full image pipeline for one article. NEVER raises.

        Returns:
          {
            "featured_image_url": str | "",
            "featured_image_alt": str | "",
            "html":               str | ""   — HTML with inline images, or ""
                                              if nothing was inserted
            "warnings":           [str]
          }
        """
        title    = content_row.get("Title",   "")
        keyword  = content_row.get("Keyword", title)
        section  = content_row.get("Section", "")
        summary  = content_row.get("Summary", "")
        slug     = re.sub(r"[^a-z0-9-]", "",
                          (optimizer_result.get("url_slug") or keyword)
                          .lower().replace(" ", "-")) or "article"
        html     = optimizer_result.get("html", "")
        warnings = []

        print(f"\n🎨 Image Agent running for: {title}")

        scene_context = (
            f"Article topic: {title}. Theme: {section or 'women’s wellness'}. "
            f"Summary: {summary or keyword}."
        )

        # ── 1. Featured (hero) image ──────────────────────────────────────
        hero_prompt = (
            f"{ART_DIRECTION}\n\nScene: an editorial lifestyle photograph that "
            f"visually represents this article for a woman in perimenopause. "
            f"{scene_context} Wide 16:9 landscape composition with gentle "
            f"negative space. The mood should feel real, warm and capable — "
            f"never sad, never clinical, never a spa."
        )
        featured_url = ""
        featured_alt = f"{keyword} — {self.brand_name}"
        files_upload_ok = True   # flips False if Shopify Files rejects uploads
        hero_bytes, hero_fal_url = self.generate_with_qa(
            hero_prompt, title, "featured hero image", image_size="landscape_16_9"
        )
        if hero_bytes:
            featured_url = self.upload_image(
                hero_bytes, f"{slug}-hero.jpg", featured_alt,
                mime_type="image/jpeg",
            ) or ""
            if not featured_url:
                files_upload_ok = False
            if not featured_url and hero_fal_url:
                # No write_files scope? articleCreate ingests external URLs
                # into Shopify's own CDN — pass the fal.media URL directly.
                print("   🔁 Shopify Files upload unavailable — using fal.media "
                      "URL (Shopify ingests it on articleCreate).")
                featured_url = hero_fal_url
        if not featured_url:
            warnings.append("Hero image failed — Publisher will use stock fallback.")

        # ── 2. Inline lifestyle image ─────────────────────────────────────
        inline_url = ""
        inline_alt = f"{keyword} — daily routine, {self.brand_name}"
        inline_prompt = (
            f"{ART_DIRECTION}\n\nScene: a second, DIFFERENT editorial photograph "
            f"for the body of the same article — a close, intimate detail moment "
            f"(hands, objects, a quiet routine) rather than a full portrait. "
            f"{scene_context} Landscape 4:3 composition."
        )
        inline_bytes = None
        if not files_upload_ok:
            # Don't spend a generation on an image we can't host —
            # body images need Shopify Files (write_files scope).
            print("   ⏭️  Skipping inline image — Shopify Files upload unavailable.")
        else:
            inline_bytes, _ = self.generate_with_qa(
                inline_prompt, title, "inline body image", image_size="landscape_4_3"
            )
        if inline_bytes:
            inline_url = self.upload_image(
                inline_bytes, f"{slug}-inline.jpg", inline_alt,
                mime_type="image/jpeg",
            ) or ""
        if not inline_url:
            # NOTE: no fal.media fallback here on purpose — body <img> tags
            # need a durable host (fal URLs are not guaranteed to persist),
            # so inline photos require the write_files scope.
            warnings.append("Inline image failed — article body has no photo.")

        # ── 3. Infographic ────────────────────────────────────────────────
        infographic_url = ""
        infographic_alt = f"{keyword} infographic — key facts by {self.brand_name}"
        data = None
        if not files_upload_ok:
            print("   ⏭️  Skipping infographic — Shopify Files upload unavailable.")
        else:
            data = self.extract_infographic_data(html, title, keyword)
        if data:
            try:
                info_bytes = self.render_infographic(data)
                infographic_url = self.upload_image(
                    info_bytes, f"{slug}-infographic.png", infographic_alt
                ) or ""
            except Exception as e:
                print(f"   ⚠️  Infographic render failed: {e}")
        if not infographic_url:
            warnings.append("Infographic failed — article has no infographic.")

        # ── 4. Insert into HTML ───────────────────────────────────────────
        new_html = ""
        if html and (inline_url or infographic_url):
            try:
                new_html = self.insert_images(
                    html, inline_url, inline_alt,
                    infographic_url, infographic_alt
                )
            except Exception as e:
                print(f"   ⚠️  HTML image insertion failed: {e}")
                warnings.append(f"Image insertion failed: {e}")
                new_html = ""

        print(f"   🖼️  Hero: {'✅' if featured_url else '❌'}  "
              f"Inline: {'✅' if inline_url else '❌'}  "
              f"Infographic: {'✅' if infographic_url else '❌'}")

        return {
            "featured_image_url": featured_url,
            "featured_image_alt": featured_alt,
            "html":               new_html,
            "warnings":           warnings,
        }


# ─────────────────────────────────────────────
# STANDALONE TEST — renders a sample infographic locally (no API calls)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = ImageAgent()
    png = agent.render_infographic({
        "headline": "Why You Wake Up at 3am in Perimenopause",
        "facts": [
            "Progesterone starts declining years before your last period",
            "Cortisol naturally peaks between 2am and 4am — lower estrogen amplifies it",
            "A 2023 study of 412 women linked night waking to temperature dysregulation",
            "Cooling your room to 65°F (18°C) may support deeper sleep",
        ],
    })
    out = "infographic_test.png"
    with open(out, "wb") as f:
        f.write(png)
    print(f"✅ Test infographic written to {out}")
