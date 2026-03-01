"""
chroma_manager.py — Semantic memory for the blog content pipeline.

This file manages ChromaDB, a local vector database that stores the
'meaning' of every published blog post as a mathematical representation
called an 'embedding'.

Unlike a regular database that matches exact words, ChromaDB matches
by MEANING. So "best supplements for perimenopause" and "top vitamins
for women in menopause transition" would be detected as very similar
even though they share almost no words.

How it's used in the pipeline:
- Publisher (Agent 6) → saves every published post to ChromaDB
- Planner  (Agent 1)  → checks ChromaDB before proposing new topics
                         to avoid writing semantically duplicate content
- Optimizer (Agent 5) → queries ChromaDB to find related posts
                         for automatic internal linking

Storage:
- All data is saved locally in a folder called 'chroma_db/'
- No external API or subscription needed — runs 100% on your machine
- Each client gets their own isolated collection (no data mixing)
"""

import os
import logging
import chromadb
from chromadb.utils import embedding_functions


class ChromaManager:
    """
    Manages the ChromaDB vector database for published post memory.

    Each client (brand) gets their own ChromaDB collection so their
    content history is completely isolated from other clients.

    Collection name format: "posts_{brand_name_lowercase_no_spaces}"
    Example: "posts_glomend", "posts_brandb"
    """

    # ↑ Everything below must be indented 4 spaces to be inside the class

    def __init__(self, persist_directory="./chroma_db", brand_name="default"):
        """
        Initializes the ChromaDB client and connects to the brand's collection.

        Parameters:
        - persist_directory: Folder where ChromaDB stores its files locally.
                             Defaults to 'chroma_db/' in your project folder.
        - brand_name:        The brand name from Config_Brand. Used to name the
                             collection so each client has isolated memory.

        What happens here:
        1. Creates (or opens) the local ChromaDB storage folder.
        2. Sets up the embedding function — this converts text to numbers
           so ChromaDB can measure how similar two pieces of text are.
        3. Creates (or opens) the collection for this brand.
        """
        self.persist_directory = persist_directory
        self.brand_name        = brand_name
        self.collection_name   = f"posts_{brand_name.lower().replace(' ', '_')}"
        self.collection        = None  # safe default if init fails below

        try:
            # PersistentClient saves data to disk so it survives restarts.
            # Without this, ChromaDB would forget everything when the script stops.
            self.client = chromadb.PersistentClient(path=persist_directory)

            # The embedding function converts text into a list of numbers.
            # We use DefaultEmbeddingFunction which runs all-MiniLM-L6-v2 locally:
            # - No API key needed
            # - Downloads automatically on first use (~80MB)
            # - Fast and accurate enough for blog topic matching
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

            # Get or create the collection for this brand.
            # If it already exists, it just opens it — no data is lost.
            self.collection = self.client.get_or_create_collection(
                name               = self.collection_name,
                embedding_function = self.embedding_fn,
                metadata           = {"hnsw:space": "cosine"}
                # cosine distance = measures angle between vectors
                # 0 = identical meaning, 1 = completely different
            )

            count = self.collection.count()
            print(f"   🧠 ChromaDB ready: '{self.collection_name}' ({count} posts in memory)")

        except Exception as e:
            logging.error(f"[ChromaDB] Initialization failed: {e}")
            print(f"   ⚠️ ChromaDB init failed: {e}")
            self.collection = None  # pipeline continues without ChromaDB


    # ─────────────────────────────────────────────
    # SAVE — called by Publisher after publishing
    # ─────────────────────────────────────────────

    def save_post(self, post: dict) -> bool:
        """
        Saves a published post to ChromaDB memory.

        Called by Publisher (Agent 6) immediately after a post goes live
        so the Planner and Optimizer can reference it in future runs.

        Parameters (all come from the Content_Plan row + optimizer_result):
        - post: dict with these keys:
            - title           (required) → used as the main text for embedding
            - url_slug        (required) → used as the unique ID in ChromaDB
            - keyword         (optional) → primary SEO keyword
            - section         (optional) → blog section/category
            - summary         (optional) → brief description of the post
            - published_date  (optional) → date published (YYYY-MM-DD)
            - admin_url       (optional) → Shopify admin URL

        What gets embedded (converted to numbers for similarity matching):
        - A combined string of: title + keyword + summary
        - This gives ChromaDB enough context to detect semantic duplicates

        Returns True if saved successfully, False if it failed.
        """
        if not self.collection:
            return False

        title    = post.get("title",    "").strip()
        url_slug = post.get("url_slug", "").strip()

        if not title or not url_slug:
            logging.warning("[ChromaDB] save_post: 'title' and 'url_slug' are required.")
            return False

        # The text that gets converted to an embedding (the "meaning" representation).
        # Combining title + keyword + summary gives a richer semantic fingerprint.
        document_text = " | ".join(filter(None, [
            title,
            post.get("keyword", ""),
            post.get("summary", "")
        ]))

        # Metadata is stored alongside the embedding for retrieval and filtering
        metadata = {
            "title":          title,
            "keyword":        post.get("keyword",        ""),
            "section":        post.get("section",        ""),
            "summary":        post.get("summary",        ""),
            "published_date": post.get("published_date", ""),
            "admin_url":      post.get("admin_url",      ""),
            "brand":          self.brand_name,
        }

        try:
            # upsert = insert if new, update if url_slug already exists.
            # This prevents duplicate entries if the pipeline runs twice.
            self.collection.upsert(
                ids       = [url_slug],
                documents = [document_text],
                metadatas = [metadata]
            )
            print(f"   🧠 ChromaDB: saved '{title}' to memory.")
            return True

        except Exception as e:
            logging.error(f"[ChromaDB] save_post failed for '{title}': {e}")
            print(f"   ⚠️ ChromaDB save failed: {e}")
            return False


    # ─────────────────────────────────────────────
    # CHECK DUPLICATES — called by Planner
    # ─────────────────────────────────────────────

    def is_duplicate(self, title: str, keyword: str = "", threshold: float = 0.85) -> dict:
        """
        Checks if a proposed topic is semantically too similar to an existing post.

        Called by Planner (Agent 1) before adding a new topic to the Sheet.
        Prevents writing content that's essentially the same as something
        already published, even if the title wording is different.

        Parameters:
        - title:     The proposed post title to check.
        - keyword:   The primary keyword (adds context to the check).
        - threshold: Similarity score above which a topic is considered a
                     duplicate. Range: 0.0 (no match) to 1.0 (identical).
                     Default 0.85 = 85% similar → flag as duplicate.

        Returns:
        {
            "is_duplicate":  True/False,
            "score":         0.92,
            "matched_title": "Existing Title",
            "matched_slug":  "existing-slug"
        }
        """
        if not self.collection or self.collection.count() == 0:
            # No memory yet (first run) or ChromaDB unavailable — allow the topic
            return {"is_duplicate": False, "score": 0.0, "matched_title": "", "matched_slug": ""}

        query_text = " | ".join(filter(None, [title, keyword]))

        try:
            results = self.collection.query(
                query_texts = [query_text],
                n_results   = 1,
                include     = ["metadatas", "distances"]
            )

            if not results["distances"] or not results["distances"][0]:
                return {"is_duplicate": False, "score": 0.0, "matched_title": "", "matched_slug": ""}

            # ChromaDB returns DISTANCE (0 = identical, 1 = completely different).
            # We convert to SIMILARITY (1 = identical, 0 = completely different).
            distance      = results["distances"][0][0]
            similarity    = 1.0 - distance
            matched_meta  = results["metadatas"][0][0] if results["metadatas"][0] else {}
            matched_title = matched_meta.get("title",   "")
            matched_slug  = matched_meta.get("keyword", "")
            is_dup        = similarity >= threshold

            if is_dup:
                print(f"   🧠 ChromaDB: '{title}' is {similarity:.0%} similar to '{matched_title}' — flagged as duplicate.")
            else:
                print(f"   🧠 ChromaDB: '{title}' is {similarity:.0%} similar to nearest post — OK to write.")

            return {
                "is_duplicate":  is_dup,
                "score":         round(similarity, 4),
                "matched_title": matched_title,
                "matched_slug":  matched_slug
            }

        except Exception as e:
            logging.error(f"[ChromaDB] is_duplicate check failed: {e}")
            return {"is_duplicate": False, "score": 0.0, "matched_title": "", "matched_slug": ""}


    # ─────────────────────────────────────────────
    # FIND RELATED — called by Optimizer
    # ─────────────────────────────────────────────

    def find_related_posts(self, title: str, keyword: str = "", n: int = 5) -> list[dict]:
        """
        Finds the N most semantically related posts to a given topic.

        Called by Optimizer (Agent 5) to find internal link candidates.
        Instead of matching exact words, it finds posts that cover related
        concepts even if they use different terminology.

        Returns a list of dicts:
        [{"title": "...", "url_slug": "...", "keyword": "...", "score": 0.78}]

        Returns an empty list if ChromaDB is unavailable or has no posts.
        """
        if not self.collection or self.collection.count() == 0:
            return []

        query_text = " | ".join(filter(None, [title, keyword]))

        try:
            # Request n+1 results in case the current post itself is in memory
            results = self.collection.query(
                query_texts = [query_text],
                n_results   = min(n + 1, self.collection.count()),
                include     = ["metadatas", "distances"]
            )

            related = []
            for meta, distance in zip(
                results["metadatas"][0],
                results["distances"][0]
            ):
                similarity = 1.0 - distance

                # Skip the current post itself (exact title match)
                if meta.get("title", "").lower() == title.lower():
                    continue

                # Only include posts with meaningful similarity (>30%)
                if similarity < 0.30:
                    continue

                related.append({
                    "title":    meta.get("title",   ""),
                    "url_slug": meta.get("keyword", ""),
                    "keyword":  meta.get("keyword", ""),
                    "section":  meta.get("section", ""),
                    "score":    round(similarity, 4)
                })

                if len(related) >= n:
                    break

            return related

        except Exception as e:
            logging.error(f"[ChromaDB] find_related_posts failed: {e}")
            return []


    # ─────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────

    def get_all_posts(self) -> list[dict]:
        """
        Returns all posts currently stored in ChromaDB memory.
        Useful for debugging, dashboards, or bulk operations.
        """
        if not self.collection or self.collection.count() == 0:
            return []
        try:
            results = self.collection.get(include=["metadatas"])
            return results.get("metadatas", [])
        except Exception as e:
            logging.error(f"[ChromaDB] get_all_posts failed: {e}")
            return []

    def get_count(self) -> int:
        """Returns the number of posts stored in ChromaDB memory."""
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

    def delete_post(self, url_slug: str) -> bool:
        """
        Removes a post from ChromaDB memory by its URL slug.

        Use this if you delete a post from Shopify and want to remove
        it from memory so the Planner can plan a similar topic again.
        """
        if not self.collection:
            return False
        try:
            self.collection.delete(ids=[url_slug])
            print(f"   🧠 ChromaDB: removed '{url_slug}' from memory.")
            return True
        except Exception as e:
            logging.error(f"[ChromaDB] delete_post failed for '{url_slug}': {e}")
            return False
