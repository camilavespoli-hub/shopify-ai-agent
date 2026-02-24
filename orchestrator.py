import os
import json
import time
import logging
import requests
import gspread
from dotenv import load_dotenv

from database_manager  import DatabaseManager
from planner    import PlannerAgent
from researcher import ResearcherAgent
from writer     import WriterAgent
from reviewer   import ReviewerAgent
from optimizer  import OptimizerAgent
from publisher  import PublisherAgent

load_dotenv()

logging.basicConfig(
    filename  = "system.log",
    level     = logging.INFO,
    format    = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt   = "%Y-%m-%d %H:%M:%S"
)

COL = {
    "RowID":             1,   # A
    "Status":            2,   # B
    "Title":             3,   # C
    "Section":           4,   # D
    "Keyword":           5,   # E
    "SecondaryKeywords": 6,   # F
    "Summary":           7,   # G
    "ScheduledDate":     8,   # H
    "ScheduledTime":     9,   # I
    "Reviewer_Notes":    10,  # J
    "Research_Brief":    11,  # K
    "Draft_Content":     12,  # L
    "WordCount":         13,  # M
    "Optimized_Draft":   14,  # N
    "Schema":            15,  # O
    "MetaTitle":         16,  # P
    "MetaDescription":   17,  # Q
    "UrlSlug":           18,  # R
    "AdminURL":          19,  # S
    "Published_Status":  20,  # T
    "Visibility":        21,  # U
}


class ContentOrchestrator:
    def __init__(self, sheet_name="Blog_agent_ai"):
        print("🤖 Initializing Orchestrator...")
        try:
            self.client = gspread.service_account(filename="service_account.json")
            self.sh     = self.client.open(sheet_name)
            print(f"✅ Connected to Google Sheet: '{sheet_name}'")
            self.db = DatabaseManager()
        except Exception as e:
            print(f"🚨 Connection Failed: {e}")
            print("Verify the sheet is shared with the service_account.json email.")
            exit()
        self.config = {}

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _log(self, level, agent, title, status, note=""):
        msg = f"[{agent}] '{title}' → {status}"
        if note:
            msg += f" | {note}"
        getattr(logging, level, logging.info)(msg)
        print(msg)

    def _update_cells(self, worksheet, row_index, updates: dict):
        cell_list = []
        for col_key, value in updates.items():
            col_num = COL.get(col_key)
            if not col_num:
                print(f"   ⚠️ _update_cells: unknown column key '{col_key}' — skipped.")
                continue
            cell       = worksheet.cell(row_index, col_num)
            cell.value = str(value) if value is not None else ""
            cell_list.append(cell)
        if cell_list:
            worksheet.update_cells(cell_list)

    def _safe_json_parse(self, raw_value, fallback=None):
        if not raw_value or not isinstance(raw_value, str):
            return fallback if fallback is not None else {}
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return fallback if fallback is not None else {}

    def _planner_row_to_sheet_list(self, row_dict):
        return [
            "",
            row_dict.get("Status",             "Pending Approval"),
            row_dict.get("Title",              ""),
            row_dict.get("Section",            ""),
            row_dict.get("Keyword",            ""),
            row_dict.get("SecondaryKeywords",  ""),
            row_dict.get("Summary",            ""),
            row_dict.get("ScheduledDate",      ""),
            row_dict.get("ScheduledTime",      "09:00 AM ET"),
            row_dict.get("TrendScore",         ""),
            row_dict.get("ReviewerNotes",      ""),
            "", "", "", "", "", "", "", "", "", "",
        ]

    # ─────────────────────────────────────────────
    # SYSTEM STATUS
    # ─────────────────────────────────────────────

    def check_system_status(self):
        try:
            ws      = self.sh.worksheet("Config_System")
            records = ws.get_all_records()
            status  = "PAUSED"
            for row in records:
                if row.get("Setting_Name") == "System_Status":
                    status = row.get("Setting_Value", "PAUSED")
                    break
            print(f"⚙️ System Status: {status}")
            if status == "ACTIVE":
                return True
            print("🛑 STOPPED." if status == "STOPPED" else "⏸️ PAUSED.")
            return False
        except Exception as e:
            print(f"⚠️ Error reading Config_System: {e}")
            return False

    # ─────────────────────────────────────────────
    # CONFIGURATION LOADER
    # ─────────────────────────────────────────────

    def load_configurations(self):
        print("📂 Loading Configurations...")
        try:
            brand_rows = self.sh.worksheet("Config_Brand").get_all_records()
            brand_flat = {
                row["Field Name"]: row["Value"]
                for row in brand_rows
                if row.get("Field Name")
            }
            sections = self.sh.worksheet("Config_Sections").get_all_records()

            try:
                product_rows = self.sh.worksheet("Config_Products").get_all_records()
            except Exception:
                product_rows = []
                print("   ⚠️ Config_Products tab not found.")

            try:
                cadence_rows   = self.sh.worksheet("Config_Cadence").get_all_records()
                active_cadence = next(
                    (r for r in cadence_rows if str(r.get("Active","")).upper()=="Y"), {}
                )
            except Exception:
                active_cadence = {}
                print("   ⚠️ Config_Cadence tab not found.")

            try:
                planner_rows = self.sh.worksheet("Config_Planner").get_all_records()
                planner_cfg  = {
                    row["Field_Name"]: row["Value"]
                    for row in planner_rows if row.get("Field_Name")
                }
            except Exception:
                planner_cfg = {}

            self.config = {
                "brand":    brand_flat,
                "planner":  planner_cfg,
                "sections": sections,
                "products": product_rows,
                "cadence":  active_cadence,
            }

            required = [
                "brand_voice_summary", "fda_disclaimer_text",
                "default_word_count_min", "default_word_count_max", "Source_Policy",
            ]
            missing = [f for f in required if not brand_flat.get(f)]
            if missing:
                print(f"   ⚠️ Missing Config_Brand fields: {missing}")

            print(f"   Brand    : {brand_flat.get('brand_name', '⚠️ not set')}")
            print(f"   Sections : {len(sections)}")
            print(f"   Products : {len(product_rows)}")
            print(f"   Cadence  : {active_cadence.get('PeriodType', '⚠️ not set')}")
            print(f"   Gemini   : {'✅' if os.getenv('GOOGLE_API_KEY')    else '❌ MISSING'}")
            print(f"   Shopify  : {'✅' if os.getenv('SHOPIFY_SHOP')       else '❌ MISSING'}")
            print(f"   Pexels   : {'✅' if os.getenv('PEXELS_API_KEY')     else '⚠️ missing'}")
            print(f"   Telegram : {'✅' if os.getenv('TELEGRAM_BOT_TOKEN') else '⚠️ missing'}")
            return True

        except Exception as e:
            print(f"🚨 Configuration Load Error: {e}")
            return False

    # ─────────────────────────────────────────────
    # NOTIFICATION
    # ─────────────────────────────────────────────

    def send_telegram_alert(self, title, admin_url=""):
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            print("   ⚠️ Telegram credentials missing — skipping.")
            return
        message = (
            f"🚀 *New Blog Draft Created!*\n\n"
            f"*Title:* {title}\n"
            f"*Review:* {admin_url or 'Check Shopify Admin'}"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10
            )
            print("   📲 Telegram notification sent.")
        except Exception as e:
            print(f"   ⚠️ Telegram alert failed: {e}")

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────

    def run(self, start_from_agent=1):
        if not self.check_system_status():
            return
        if not self.load_configurations():
            return

        print("\n🚀 System is GO. Starting pipeline...\n")

        plan_sheet     = self.sh.worksheet("Content_Plan")
        source_policy  = self.config["brand"].get("Source_Policy", "")
        valid_sections = [s["Name"] for s in self.config["sections"]]

        if not source_policy:
            print("🛑 Stopped: 'Source_Policy' missing from Config_Brand.")
            return

        planner    = PlannerAgent(config=self.config)
        researcher = ResearcherAgent(config=self.config)
        writer     = WriterAgent(config=self.config)
        reviewer   = ReviewerAgent(config=self.config)
        optimizer  = OptimizerAgent(config=self.config)
        publisher  = PublisherAgent(config=self.config, optimizer_agent=optimizer)

        # ══════════════════════════════════════════════════════════════
        # AGENT 1: PLANNER
        # ══════════════════════════════════════════════════════════════
        if start_from_agent <= 1:
            print("─" * 60)
            print("🧠 AGENT 1: PLANNER")
            try:
                existing_records = plan_sheet.get_all_records()

                # DEBUG — borra después
                for i, r in enumerate(existing_records, start=2):
                    if not r.get("Title"):
                        continue
                    print(f"   Row {i}: "
                        f"Status='{r.get('Status')}' | "
                        f"OptDraft={'YES' if r.get('Optimized_Draft') else 'EMPTY'} | "
                        f"PubStatus='{r.get('Published_Status')}'")
                    #hasta acá
                existing_titles  = [r.get("Title","") for r in existing_records if r.get("Title")]

                plan_result = planner.plan_next_posts(
                    sections_config   = self.config["sections"][:1],
                    existing_titles   = existing_titles,
                    config            = self.config,
                    posts_per_section = 1
                )

                if plan_result["status"] == "BLOCKED":
                    self._log("error", "Planner", "batch", "BLOCKED",
                            str(plan_result.get("errors")))
                else:
                    new_rows = plan_result.get("new_rows", [])
                    if new_rows:
                        plan_sheet.append_rows(
                            [self._planner_row_to_sheet_list(r) for r in new_rows]
                        )
                        self._log("info", "Planner", "batch", f"Added {len(new_rows)} topic(s)")
                        self.db.log_task("Planner", "batch", "SUCCESS",
                                        f"Added {len(new_rows)} topic(s).")
                    else:
                        print("   ⚠️ Planner generated 0 topics.")
                    for w in plan_result.get("warnings", []):
                        self._log("warning", "Planner", "batch", "WARNING", w)

            except Exception as e:
                self._log("error", "Planner", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════
        # AGENT 2: RESEARCHER
        # ══════════════════════════════════════════════════════════════
        if start_from_agent <= 2:
            print("\n" + "─" * 60)
            print("🔬 AGENT 2: RESEARCHER")
            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    if row.get("Status") != "Pending Approval":
                        continue
                    if row.get("Research_Brief"):
                        continue

                    title          = row.get("Title", "")
                    research_result = researcher.research_topic(
                        content_row    = row,
                        valid_sections = valid_sections,
                        source_policy  = source_policy
                    )
                    status     = research_result.get("status", "")
                    brief_json = json.dumps(research_result)

                    if status == "BLOCKED":
                        self._update_cells(plan_sheet, index, {
                            "Reviewer_Notes": "; ".join(research_result.get("errors", []))
                        })
                        self._log("error", "Researcher", title, "BLOCKED",
                                str(research_result.get("errors")))
                        self.db.log_task("Researcher", title, "BLOCKED",
                                        str(research_result.get("errors")))

                    elif "insufficient sources" in status:
                        self._update_cells(plan_sheet, index, {
                            "Status":         "Needs Review",
                            "Research_Brief": brief_json,
                            "Reviewer_Notes": "Needs Review: insufficient sources"
                        })
                        self._log("warning", "Researcher", title,
                                "Needs Review — insufficient sources")
                        self.db.log_task("Researcher", title, "NEEDS REVIEW",
                                        "Insufficient sources.")

                    else:
                        self._update_cells(plan_sheet, index, {
                            "Research_Brief": brief_json,
                            "Reviewer_Notes": "; ".join(research_result.get("warnings", []))
                        })
                        self._log("info", "Researcher", title, status)
                        self.db.log_task("Researcher", title, "SUCCESS", status)
                        print("🛑 Test Mode: stopping Researcher after 1 run.")
                        time.sleep(10)
                        break

                    time.sleep(10)

            except Exception as e:
                self._log("error", "Researcher", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════
        # AGENT 3 + 4: WRITER → REVIEWER LOOP (max 3 attempts)
        # ══════════════════════════════════════════════════════════════
        if start_from_agent <= 3:
            print("\n" + "─" * 60)
            print("✍️  AGENT 3: WRITER  +  🛡️ AGENT 4: REVIEWER (auto-retry loop)")
            MAX_RETRIES = 3

            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    if row.get("Status") != "Pending Approval":
                        continue
                    if not row.get("Research_Brief"):
                        continue
                    if row.get("Draft_Content") and row.get("Reviewer_Notes"):
                        continue

                    research_result = self._safe_json_parse(
                        row.get("Research_Brief"), fallback={}
                    )
                    if not research_result:
                        print(f"   ⚠️ Row {index}: could not parse Research_Brief — skipping.")
                        continue

                    research_status = research_result.get("status", "")
                    if "BLOCKED" in research_status or "insufficient" in research_status:
                        print(f"   ⚠️ Row {index}: bad research status — skipping.")
                        continue

                    title          = row.get("Title", "")
                    previous_draft = ""
                    required_fixes = []
                    final_writer   = None
                    final_review   = None

                    for attempt in range(1, MAX_RETRIES + 1):
                        print(f"\n   🔄 Attempt {attempt}/{MAX_RETRIES}: Writing...")

                        writer_result = writer.write_draft(
                            content_row     = row,
                            research_result = research_result,
                            config          = self.config,
                            previous_draft  = previous_draft,
                            required_fixes  = required_fixes
                        )

                        if writer_result["status"] == "BLOCKED":
                            self._log("error", "Writer", title, "BLOCKED",
                                    str(writer_result.get("errors")))
                            self.db.log_task("Writer", title, "BLOCKED",
                                            str(writer_result.get("errors")))
                            break

                        if writer_result["status"] == "HARD_FAIL":
                            self._log("warning", "Writer", title, "HARD_FAIL",
                                    str(writer_result.get("violations")))
                            self.db.log_task("Writer", title, "HARD_FAIL",
                                            str(writer_result.get("violations")))
                            break

                        current_draft = writer_result.get("html", "")
                        print(f"   ✍️  Draft written ({writer_result.get('word_count',0)} words). Reviewing...")

                        review_result = reviewer.review_draft(
                            content_row     = row,
                            draft_text      = current_draft,
                            research_result = research_result,
                            config          = self.config
                        )

                        review_status = review_result.get("status", "")
                        print(f"   🛡️  Review: {review_status}")

                        if review_status in ("PASS", "PASS_WITH_NOTES"):
                            final_writer = writer_result
                            final_review = review_result
                            print(f"   ✅ Passed on attempt {attempt}!")
                            break

                        required_fixes = review_result.get("required_fixes", [])
                        previous_draft = current_draft
                        print(f"   ↩️  {len(required_fixes)} fix(es) sent back to Writer:")
                        for fix in required_fixes:
                            print(f"      - {fix}")

                        if attempt == MAX_RETRIES:
                            final_writer = writer_result
                            final_review = review_result
                            print("   ⚠️ Max retries reached — saving draft as 'Needs Review'.")

                    if final_writer and final_review:
                        review_status  = final_review.get("status", "FAIL")
                        summary_note   = final_review.get("reviewer_summary", "")
                        new_status     = "Content Approved" if review_status in (
                            "PASS", "PASS_WITH_NOTES"
                        ) else "Needs Review"

                        self._update_cells(plan_sheet, index, {
                            "Draft_Content":  final_writer.get("html",       ""),
                            "WordCount":      final_writer.get("word_count", 0),
                            "Status":         new_status,
                            "Reviewer_Notes": summary_note
                        })
                        self._log("info", "Writer+Reviewer", title, new_status,
                                f"Words: {final_writer.get('word_count',0)}")
                        self.db.log_task("Writer+Reviewer", title,
                                        "SUCCESS" if new_status == "Content Approved"
                                        else "NEEDS_REVIEW", summary_note)
                        print("🛑 Test Mode: stopping after 1 article.")
                        time.sleep(10)
                        break

            except Exception as e:
                self._log("error", "Writer+Reviewer", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════
        # AGENT 5: OPTIMIZER
        # ══════════════════════════════════════════════════════════════
        if start_from_agent <= 5:
            print("\n" + "─" * 60)
            print("⚙️ AGENT 5: OPTIMIZER")
            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    if row.get("Status") != "Content Approved":
                        continue
                    if not row.get("Draft_Content"):
                        continue
                    if row.get("Optimized_Draft"):
                        continue

                    title           = row.get("Title", "")
                    reviewer_result = {
                        "status":           "PASS",
                        "reviewer_summary": row.get("Reviewer_Notes", "Approved")
                        }
                    optimizer_result = optimizer.optimize_draft(
                        content_row     = row,
                        draft_text      = row.get("Draft_Content", ""),
                        reviewer_result = reviewer_result,
                        config          = self.config
                        )
                    status = optimizer_result.get("status", "")

                    if status == "BLOCKED":
                        self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": "; ".join(optimizer_result.get("errors", []))
                            })
                        self._log("error", "Optimizer", title, "BLOCKED",
                                    str(optimizer_result.get("errors")))
                        self.db.log_task("Optimizer", title, "BLOCKED",
                                            str(optimizer_result.get("errors")))    
                    else:
                        # Non-blocking warnings — informational only, don't stop pipeline
                        non_blocking_warnings = [
                            "Internal link inventory unavailable",
                            "schema_type missing",
                        ]
                        blocking_warnings = [
                            w for w in optimizer_result.get("warnings", [])
                            if not any(nb in w for nb in non_blocking_warnings)
                        ]
                        new_status = "Ready to Publish" if not blocking_warnings else "Needs Review"

                        self._update_cells(plan_sheet, index, {
                            "Status":          new_status,
                                "Optimized_Draft": optimizer_result.get("html",             ""),
                                "Schema":          optimizer_result.get("schema",           ""),
                                "MetaTitle":       optimizer_result.get("meta_title",       ""),
                                "MetaDescription": optimizer_result.get("meta_description", ""),
                                "UrlSlug":         optimizer_result.get("url_slug",         ""),
                                "WordCount":       optimizer_result.get("word_count",       ""),
                                "Reviewer_Notes":  "; ".join(
                                    optimizer_result.get("warnings", [])
                                ) or row.get("Reviewer_Notes", "")
                            })
                        self._log("info", "Optimizer", title, new_status,
                                    f"Slug: {optimizer_result.get('url_slug','N/A')}")
                        self.db.log_task("Optimizer", title, "SUCCESS",
                                        f"Status → {new_status}")
                        print("🛑 Test Mode: stopping Optimizer after 1 run.")
                        time.sleep(10)
                        break

                        time.sleep(10)

            except Exception as e:
                self._log("error", "Optimizer", "N/A", "EXCEPTION", str(e))


        # ══════════════════════════════════════════════════════════════
        # AGENT 6: PUBLISHER
        # ══════════════════════════════════════════════════════════════
        if start_from_agent <= 6:
            print("\n" + "─" * 60)
            print("🚀 AGENT 6: PUBLISHER")
            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    if row.get("Status") != "Ready to Publish":
                        continue
                    if not row.get("Optimized_Draft"):
                        continue
                    if row.get("Published_Status"):
                        continue

                    title            = row.get("Title", "")
                    optimizer_result = {
                        "status":           "Ready for Approval",
                        "html":             row.get("Optimized_Draft",   ""),
                        "schema":           row.get("Schema",            ""),
                        "meta_title":       row.get("MetaTitle",         ""),
                        "meta_description": row.get("MetaDescription",   ""),
                        "url_slug":         row.get("UrlSlug",           ""),
                        "word_count":       row.get("WordCount",         0),
                        "warnings":         []
                    }
                    publish_result = publisher.publish_post(
                        content_row      = row,
                        optimizer_result = optimizer_result,
                        config           = self.config
                    )
                    pub_status = publish_result.get("status", "")

                    if pub_status in ("BLOCKED", "FAILED"):
                        error_note = "; ".join(
                            publish_result.get("errors",   []) +
                            publish_result.get("warnings", [])
                        )
                        self._update_cells(plan_sheet, index, {"Reviewer_Notes": error_note})
                        self._log("error", "Publisher", title, pub_status, error_note)
                        self.db.log_task("Publisher", title, pub_status, error_note)
                    else:
                        admin_url    = publish_result.get("admin_url", "")
                        warning_note = "; ".join(publish_result.get("warnings", []))
                        self._update_cells(plan_sheet, index, {
                            "Status":           "Live",
                            "Published_Status": "Draft Created",
                            "AdminURL":         admin_url,
                            "Reviewer_Notes":   warning_note
                        })
                        self._log("info", "Publisher", title, "Draft Created", admin_url)
                        self.db.log_task("Publisher", title, "SUCCESS", admin_url)
                        self.send_telegram_alert(title, admin_url=admin_url)
                        print("🛑 Test Mode: stopping Publisher after 1 run.")
                        break

            except Exception as e:
                self._log("error", "Publisher", "N/A", "EXCEPTION", str(e))

        print("\n✅ Pipeline run complete.")
        logging.info("Pipeline run complete.")


# ── Execution ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = ContentOrchestrator(sheet_name="Blog_agent_ai")
    bot.run(start_from_agent=5)