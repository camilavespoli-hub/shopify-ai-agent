import os
import json
import time
import logging
import requests
import gspread
from dotenv import load_dotenv

from database_manager import DatabaseManager
from planner          import PlannerAgent
from researcher       import ResearcherAgent
from writer           import WriterAgent
from reviewer         import ReviewerAgent
from optimizer        import OptimizerAgent
from publisher        import PublisherAgent

try:
    from chroma_manager import ChromaManager
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  chroma_manager not found — duplicate detection disabled.")

load_dotenv()

logging.basicConfig(
    filename = "system.log",
    level    = logging.INFO,
    format   = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S"
)

# ── Column mapping (single source of truth) ────────────────────────────────────
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
    "BlogType":          22,  # V
    "TopicCluster":      23,  # W
}


class ContentOrchestrator:

    def __init__(self, sheet_name="Blog_agent_ai"):
        print("🤖 Initializing Orchestrator...")

        self.IS_AWS   = os.getenv("AWS_EXECUTION_ENV") is not None
        self.DATA_DIR = "/mnt/efs" if self.IS_AWS else "."

        try:
            self.client = gspread.service_account(filename="service_account.json")
            self.sh     = self.client.open(sheet_name)
            print(f"✅ Connected to Google Sheet: '{sheet_name}'")
        except Exception as e:
            print(f"🚨 Google Sheets connection failed: {e}")
            print("Verify the sheet is shared with the service_account.json email.")
            exit()

        self.db     = DatabaseManager(db_path=os.path.join(self.DATA_DIR, "pipeline.db"))
        self.chroma = None
        self.config = {}

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

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
                print(f"   ⚠️  _update_cells: unknown column key '{col_key}' — skipped.")
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

    def _get_rule(self, agent_name: str, rule_key: str, default=None):
        val = (
            self.config
            .get("agent_rules", {})
            .get(agent_name, {})
            .get(rule_key, default)
        )
        if isinstance(val, str):
            if val.upper() == "TRUE":  return True
            if val.upper() == "FALSE": return False
            try:    return int(val)
            except (ValueError, TypeError): pass
            try:    return float(val)
            except (ValueError, TypeError): pass
        return val

    def _get_system_setting(self, key: str, default=None):
        val = self.config.get("system", {}).get(key, default)
        if isinstance(val, str):
            if val.upper() == "TRUE":  return True
            if val.upper() == "FALSE": return False
            try:    return int(val)
            except (ValueError, TypeError): pass
        return val

    def _planner_row_to_sheet_list(self, row_dict):
        return [
            "",                                                       # A  RowID (auto)
            row_dict.get("Status",            "Pending Approval"),    # B
            row_dict.get("Title",             ""),                    # C
            row_dict.get("Section",           ""),                    # D
            row_dict.get("Keyword",           ""),                    # E
            row_dict.get("SecondaryKeywords", ""),                    # F
            row_dict.get("Summary",           ""),                    # G
            row_dict.get("ScheduledDate",     ""),                    # H
            row_dict.get("ScheduledTime",     "09:00 AM ET"),         # I
            "",   # J  Reviewer_Notes   — empty at planning
            "",   # K  Research_Brief   — empty at planning
            "",   # L  Draft_Content    — empty
            "",   # M  WordCount        — empty
            "",   # N  Optimized_Draft  — empty
            "",   # O  Schema           — empty
            "",   # P  MetaTitle        — empty
            "",   # Q  MetaDescription  — empty
            "",   # R  UrlSlug          — empty
            "",   # S  AdminURL         — empty
            "",   # T  Published_Status — empty
            "",   # U  Visibility       — empty
            row_dict.get("BlogType",     "educational"),              # V
            row_dict.get("TopicCluster", ""),                         # W
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM STATUS
    # ─────────────────────────────────────────────────────────────────────────

    def check_system_status(self):
        try:
            ws      = self.sh.worksheet("Config_System")
            records = ws.get_all_records()
            status  = "PAUSED"
            for row in records:
                if row.get("Setting_Name") == "System_Status":
                    status = str(row.get("Setting_Value", "PAUSED")).upper()
                    break
            print(f"⚙️  System Status: {status}")
            if status == "ACTIVE":
                return True
            print("🛑 STOPPED." if status == "STOPPED" else "⏸️  PAUSED.")
            return False
        except Exception as e:
            print(f"⚠️  Error reading Config_System: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION LOADER
    # ─────────────────────────────────────────────────────────────────────────

    def load_configurations(self):
        print("📂 Loading Configurations...")
        try:
            brand_rows = self.sh.worksheet("Config_Brand").get_all_records()
            brand_flat = {
                row["Field Name"]: row["Value"]
                for row in brand_rows
                if row.get("Field Name")
            }

            try:
                system_rows = self.sh.worksheet("Config_System").get_all_records()
                system_flat = {
                    row["Setting_Name"]: row["Setting_Value"]
                    for row in system_rows
                    if row.get("Setting_Name")
                }
            except Exception:
                system_flat = {}
                print("   ⚠️  Config_System tab not found.")

            try:
                rules_rows  = self.sh.worksheet("Agent_Rules").get_all_records()
                agent_rules = {}
                for row in rules_rows:
                    agent = row.get("agent_name", "").strip()
                    key   = row.get("rule_key",   "").strip()
                    val   = row.get("rule_value",  "")
                    if agent and key:
                        agent_rules.setdefault(agent, {})[key] = val
            except Exception:
                agent_rules = {}
                print("   ⚠️  Agent_Rules tab not found.")

            try:
                prompt_rows = self.sh.worksheet("Prompts").get_all_records()
                prompts = {}
                for row in prompt_rows:
                    agent = row.get("agent_name", "").strip()
                    key   = row.get("prompt_key",  "").strip()
                    val   = row.get("prompt_text",  "")
                    if agent and key:
                        prompts.setdefault(agent, {})[key] = val
            except Exception:
                prompts = {}

            sections = self.sh.worksheet("Config_Sections").get_all_records()

            try:
                product_rows = self.sh.worksheet("Config_Products").get_all_records()
            except Exception:
                product_rows = []
                print("   ⚠️  Config_Products tab not found.")

            try:
                cadence_rows   = self.sh.worksheet("Config_Cadence").get_all_records()
                active_cadence = next(
                    (r for r in cadence_rows if str(r.get("Active", "")).upper() == "Y"), {}
                )
            except Exception:
                active_cadence = {}
                print("   ⚠️  Config_Cadence tab not found.")

            try:
                planner_rows = self.sh.worksheet("Config_Planner").get_all_records()
                planner_cfg  = {
                    row["Field_Name"]: row["Value"]
                    for row in planner_rows if row.get("Field_Name")
                }
            except Exception:
                planner_cfg = {}

            # ── Config_BlogTypes ──────────────────────────────────────────────
            # Stored as nested dict: { blog_type: { rule_key: rule_value } }
            try:
                blog_type_rows = self.sh.worksheet("Config_BlogTypes").get_all_records()
                blog_types = {}
                for row in blog_type_rows:
                    bt  = row.get("blog_type", "").strip().lower()
                    key = row.get("rule_key",  "").strip()
                    val = row.get("rule_value", "")
                    if bt and key:
                        if isinstance(val, str):
                            if val.upper() == "TRUE":  val = True
                            elif val.upper() == "FALSE": val = False
                        blog_types.setdefault(bt, {})[key] = val
            except Exception:
                blog_types = {}
                print("   ⚠️  Config_BlogTypes tab not found — defaults will be used.")

            self.config = {
                "brand":       brand_flat,
                "system":      system_flat,
                "agent_rules": agent_rules,
                "prompts":     prompts,
                "planner":     planner_cfg,
                "sections":    sections,
                "products":    product_rows,
                "cadence":     active_cadence,
                "blog_types":  blog_types,
            }

            required = [
                "brand_voice_summary",
                "disclaimer_text",
                "default_word_count_min",
                "default_word_count_max",
                "Source_Policy",
                "content_language",
                "industry",
                "compliance_framework",
            ]
            missing = [f for f in required if not brand_flat.get(f)]
            if missing:
                print(f"   ⚠️  Missing Config_Brand fields: {missing}")

            if CHROMA_AVAILABLE:
                try:
                    brand_name  = brand_flat.get("brand_name", "default")
                    chroma_path = os.path.join(self.DATA_DIR, "chroma_db")
                    self.chroma = ChromaManager(
                        persist_directory = chroma_path,
                        brand_name        = brand_name
                    )
                    print(f"   ChromaDB  : ✅ ({chroma_path})")
                except Exception as e:
                    self.chroma = None
                    print(f"   ChromaDB  : ⚠️  init failed — {e}")
            else:
                print("   ChromaDB  : ⚠️  disabled (chroma_manager.py not found)")

            rule_count = sum(len(v) for v in agent_rules.values())
            print(f"   Brand     : {brand_flat.get('brand_name',       '⚠️ not set')}")
            print(f"   Language  : {brand_flat.get('content_language', '⚠️ not set')}"
                  f"  | Industry: {brand_flat.get('industry', '⚠️ not set')}")
            print(f"   Sections  : {len(sections)}")
            print(f"   Products  : {len(product_rows)}")
            print(f"   BlogTypes : {len(blog_types)}")
            print(f"   Rules     : {rule_count} rule(s) loaded")
            print(f"   Gemini    : {'✅' if os.getenv('GOOGLE_API_KEY')    else '❌ MISSING'}")
            print(f"   Shopify   : {'✅' if os.getenv('SHOPIFY_SHOP')       else '❌ MISSING'}")
            print(f"   Pexels    : {'✅' if os.getenv('PEXELS_API_KEY')     else '⚠️ missing'}")
            print(f"   Telegram  : {'✅' if os.getenv('TELEGRAM_BOT_TOKEN') else '⚠️ missing'}")
            return True

        except Exception as e:
            print(f"🚨 Configuration Load Error: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # NOTIFICATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _send_telegram(self, message: str):
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            print("   ⚠️  Telegram credentials missing — skipping.")
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10
            )
            print("   📲 Telegram notification sent.")
        except Exception as e:
            print(f"   ⚠️  Telegram alert failed: {e}")

    def send_telegram_alert(self, title: str, admin_url: str = "",
                            hidden: bool = False, reason: str = ""):
        pub_prompts = self.config.get("prompts", {}).get("publisher", {})

        if hidden:
            template = pub_prompts.get("telegram_hidden_message", "")
            if template:
                message = (template
                    .replace("{title}",     title)
                    .replace("{reason}",    reason or "Red flags detected")
                    .replace("{admin_url}", admin_url or "Check Shopify Admin")
                )
            else:
                message = (
                    f"⚠️ *Post Published as HIDDEN — Needs Review*\n\n"
                    f"*Title:* {title}\n"
                    f"*Reason:* {reason or 'Red flags detected'}\n"
                    f"*Review:* {admin_url or 'Check Shopify Admin'}"
                )
        else:
            template = pub_prompts.get("telegram_message", "")
            if template:
                message = (template
                    .replace("{title}",     title)
                    .replace("{admin_url}", admin_url or "Check Shopify Admin")
                )
            else:
                message = (
                    f"🚀 *New Blog Post Published!*\n\n"
                    f"*Title:* {title}\n"
                    f"*Review:* {admin_url or 'Check Shopify Admin'}"
                )

        self._send_telegram(message)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, start_from_agent: int = 1):
        if not self.check_system_status():
            return
        if not self.load_configurations():
            return

        system_start = self._get_system_setting("Start_From_Agent", default=1)
        if start_from_agent == 1 and system_start != 1:
            start_from_agent = int(system_start)

        test_mode   = self._get_system_setting("Test_Mode", default=False)
        max_retries = int(
            self._get_rule("orchestrator", "max_retries",
                           default=self._get_system_setting("Max_Retries", default=3))
            or 3
        )
        posts_per_section = int(self._get_rule("orchestrator", "posts_per_section", default=1) or 1)
        sections_limit    = int(self._get_rule("orchestrator", "sections_limit",    default=1) or 1)

        print("\n🚀 System is GO. Starting pipeline...\n")
        if test_mode:
            print("🧪 TEST MODE ON — pipeline stops after 1 item per agent.\n")

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

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 1: PLANNER
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 1:
            print("─" * 60)
            print("🧠 AGENT 1: PLANNER")
            try:
                existing_records  = plan_sheet.get_all_records()
                existing_titles   = [r.get("Title", "") for r in existing_records if r.get("Title")]
                sections_to_plan  = self.config["sections"][:sections_limit]

                max_topic_attempts = int(self._get_rule("planner", "max_topic_attempts", default=3) or 3)
                duplicate_action   = self._get_rule("planner", "duplicate_action", default="skip")
                dup_threshold      = float(self.config["brand"].get("chroma_duplicate_threshold", 0.85))

                plan_result = planner.plan_next_posts(
                    sections_config   = sections_to_plan,
                    existing_titles   = existing_titles,
                    config            = self.config,
                    posts_per_section = posts_per_section
                )

                if plan_result["status"] == "BLOCKED":
                    self._log("error", "Planner", "batch", "BLOCKED",
                              str(plan_result.get("errors")))
                else:
                    new_rows      = plan_result.get("new_rows", [])
                    approved_rows = []

                    for proposed in new_rows:
                        title    = proposed.get("Title",   "")
                        keyword  = proposed.get("Keyword", "")
                        accepted = False

                        for attempt in range(1, max_topic_attempts + 1):
                            if self.chroma:
                                dup_check = self.chroma.is_duplicate(
                                    title     = title,
                                    keyword   = keyword,
                                    threshold = dup_threshold
                                )
                                is_dup  = dup_check.get("is_duplicate", False)
                                score   = dup_check.get("score",         0.0)
                                matched = dup_check.get("matched_title", "")
                            else:
                                is_dup = False

                            if not is_dup:
                                approved_rows.append(proposed)
                                accepted = True
                                break

                            if duplicate_action == "warn":
                                print(f"   ⚠️  '{title}' is {score:.0%} similar to '{matched}' — allowed (warn mode).")
                                approved_rows.append(proposed)
                                accepted = True
                                break

                            print(f"   🔄 '{title}' is {score:.0%} similar to '{matched}' "
                                  f"— retrying ({attempt}/{max_topic_attempts})...")
                            retry_result = planner.plan_next_posts(
                                sections_config   = sections_to_plan,
                                existing_titles   = existing_titles + [title],
                                config            = self.config,
                                posts_per_section = 1
                            )
                            retry_rows = retry_result.get("new_rows", [])
                            if retry_rows:
                                proposed = retry_rows[0]
                                title    = proposed.get("Title",   "")
                                keyword  = proposed.get("Keyword", "")

                        if not accepted:
                            print(f"   ⚠️  Could not find non-duplicate topic after "
                                  f"{max_topic_attempts} attempts — skipping.")
                            self._log("warning", "Planner", title, "SKIPPED",
                                      "Duplicate threshold exceeded after max retries.")
                            self.db.log_task("Planner", title, "SKIPPED",
                                             "Duplicate — max retries reached.")

                    if approved_rows:
                        plan_sheet.append_rows(
                            [self._planner_row_to_sheet_list(r) for r in approved_rows]
                        )
                        self._log("info", "Planner", "batch",
                                  f"Added {len(approved_rows)} topic(s)")
                        self.db.log_task("Planner", "batch", "SUCCESS",
                                         f"Added {len(approved_rows)} topic(s).")
                    else:
                        print("   ⚠️  Planner: no new topics added this run.")

                    for w in plan_result.get("warnings", []):
                        self._log("warning", "Planner", "batch", "WARNING", w)

            except Exception as e:
                self._log("error", "Planner", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 2: RESEARCHER
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 2:
            print("\n" + "─" * 60)
            print("🔬 AGENT 2: RESEARCHER")
            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    try:
                        if row.get("Status") != "Pending Approval":
                            continue
                        if row.get("Research_Brief"):
                            continue

                        title           = row.get("Title", "")
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

                            if test_mode:
                                print("🛑 Test Mode: stopping Researcher after 1 run.")
                                time.sleep(10)
                                break

                        time.sleep(10)

                    except Exception as row_e:
                        title = row.get("Title", f"Row {index}")
                        self._log("error", "Researcher", title, "ROW_EXCEPTION", str(row_e))
                        self.db.log_task("Researcher", title, "ROW_EXCEPTION", str(row_e))
                        try:
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": f"Pipeline error: {str(row_e)[:200]}"
                            })
                        except Exception:
                            pass
                        continue

            except Exception as e:
                self._log("error", "Researcher", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 3 + 4: WRITER → REVIEWER LOOP
        #
        # ✅ FIX vs original:
        #   - reviewer.review_draft()  (was review_content())
        #   - param draft_text=        (was draft_html=)
        #   - result["reviewer_summary"] (was result["notes"])
        #   - result["violations"]       (was result["red_flags"])
        #   - status in ("PASS","PASS_WITH_NOTES")  (was == "APPROVED")
        #   - skip condition: checks Draft_Content exists (was Reviewer_Notes=="APPROVED")
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 3:
            print("\n" + "─" * 60)
            print("✍️  AGENT 3: WRITER  +  🛡️ AGENT 4: REVIEWER (rewrite loop)")
            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    try:
                        if row.get("Status") != "Pending Approval":
                            continue
                        if not row.get("Research_Brief"):
                            continue
                        # ✅ FIX: skip if already written+reviewed
                        # (Status stays "Pending Approval" until THIS loop changes it)
                        if row.get("Draft_Content"):
                            continue

                        research_result = self._safe_json_parse(
                            row.get("Research_Brief"), fallback={}
                        )
                        if not research_result:
                            print(f"   ⚠️  Row {index}: could not parse Research_Brief — skipping.")
                            continue

                        title                     = row.get("Title", "")
                        best_draft                = None
                        best_reviewer_summary     = ""
                        final_status              = "hidden"  # default: hidden if never PASS
                        required_fixes_for_writer = []

                        for attempt in range(1, max_retries + 1):
                            print(f"\n   🔄 Attempt {attempt}/{max_retries}: Writing...")

                            write_result = writer.write_draft(
                                content_row     = row,
                                research_result = research_result,
                                config          = self.config,
                                previous_draft  = best_draft or "",
                                required_fixes  = required_fixes_for_writer,
                            )

                            write_status = write_result.get("status", "")

                            if write_status == "HARD_FAIL":
                                print(f"   🚨 Writer HARD_FAIL on attempt {attempt} — "
                                      f"compliance violation cannot be auto-fixed.")
                                self._log("error", "Writer", title, "HARD_FAIL",
                                          str(write_result.get("violations")))
                                self.db.log_task("Writer", title, "HARD_FAIL",
                                                 str(write_result.get("violations")))
                                break

                            if write_status == "BLOCKED":
                                print(f"   🚫 Writer BLOCKED on attempt {attempt}: "
                                      f"{write_result.get('errors')}")
                                self._log("error", "Writer", title, "BLOCKED",
                                          str(write_result.get("errors")))
                                break

                            draft = write_result.get("html", "")
                            if draft:
                                best_draft = draft

                            # ── ✅ FIX: reviewer.review_draft() + correct param/result names ──
                            print(f"   🛡️  Reviewer evaluating attempt {attempt}...")
                            review_result = reviewer.review_draft(
                                content_row     = row,
                                draft_text      = draft,           # ✅ FIX: was draft_html
                                research_result = research_result,
                                config          = self.config,
                            )

                            review_status    = review_result.get("status",           "")
                            reviewer_summary = review_result.get("reviewer_summary", "")  # ✅ FIX: was "notes"
                            required_fixes   = review_result.get("required_fixes",   [])
                            violations       = review_result.get("violations",        [])  # ✅ FIX: was "red_flags"

                            print(f"      Reviewer: {review_status}")

                            # ✅ FIX: PASS or PASS_WITH_NOTES = green flag (was == "APPROVED")
                            if review_status in ("PASS", "PASS_WITH_NOTES"):
                                best_draft            = draft
                                best_reviewer_summary = reviewer_summary
                                final_status          = "public"
                                print(f"   ✅ GREEN FLAG on attempt {attempt} — "
                                      f"queued for Live/Public.")
                                break

                            # FAIL → prepare feedback for Writer
                            best_reviewer_summary     = reviewer_summary
                            required_fixes_for_writer = list(required_fixes)

                            for v in violations:
                                fix = f"COMPLIANCE VIOLATION: {v}"
                                if fix not in required_fixes_for_writer:
                                    required_fixes_for_writer.append(fix)

                            print(f"   🔁 {len(required_fixes_for_writer)} issue(s) found "
                                  f"— rewriting with feedback...")

                            if attempt == max_retries:
                                print(f"   ⚠️  Max retries ({max_retries}) reached — "
                                      f"will publish as Live/Hidden for review.")

                        # ── Post-loop ─────────────────────────────────────────
                        if not best_draft:
                            self._update_cells(plan_sheet, index, {
                                "Status":         "Needs Review",
                                "Reviewer_Notes": (
                                    "All attempts failed (HARD_FAIL/BLOCKED) — "
                                    "manual intervention needed."
                                ),
                            })
                            self._log("error", "Writer", title, "NO_DRAFT_PRODUCED")
                            self.db.log_task("Writer", title, "NO_DRAFT_PRODUCED",
                                             "No valid draft after all retries.")
                            continue

                        self._update_cells(plan_sheet, index, {
                            "Draft_Content":  best_draft,
                            "WordCount":      str(len(best_draft.split())),
                            "Reviewer_Notes": best_reviewer_summary,
                            "Status":         "Ready_To_Publish",
                            # ✅ Publisher reads Visibility from content_row
                            "Visibility":     "hidden" if final_status == "hidden" else "",
                        })

                        status_label = (
                            "APPROVED → queued for Live/Public"
                            if final_status == "public"
                            else f"Max retries ({max_retries}) → queued for Live/Hidden"
                        )
                        self._log("info", "Writer+Reviewer", title, status_label)
                        self.db.log_task(
                            "Writer", title,
                            "APPROVED" if final_status == "public" else "MAX_RETRIES_HIDDEN",
                            status_label
                        )

                        if test_mode:
                            print("🛑 Test Mode: stopping Writer+Reviewer after 1 run.")
                            time.sleep(10)
                            break

                        time.sleep(10)

                    except Exception as row_e:
                        title = row.get("Title", f"Row {index}")
                        self._log("error", "Writer", title, "ROW_EXCEPTION", str(row_e))
                        self.db.log_task("Writer", title, "ROW_EXCEPTION", str(row_e))
                        try:
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": f"Pipeline error: {str(row_e)[:200]}"
                            })
                        except Exception:
                            pass
                        continue

            except Exception as e:
                self._log("error", "Writer", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 5: OPTIMIZER
        #
        # ✅ FIX vs original:
        #   - reads Status == "Ready_To_Publish"  (was "Content Approved" — never set)
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 5:
            print("\n" + "─" * 60)
            print("⚙️  AGENT 5: OPTIMIZER")

            non_blocking_raw      = self._get_rule(
                "orchestrator", "non_blocking_warnings",
                default="Internal link inventory unavailable,schema_type missing"
            )
            non_blocking_warnings = [
                w.strip() for w in str(non_blocking_raw).split(",") if w.strip()
            ]

            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    try:
                        # ✅ FIX: "Ready_To_Publish" is what Agent 3+4 writes
                        if row.get("Status") != "Ready_To_Publish":
                            continue
                        if not row.get("Draft_Content"):
                            continue
                        if row.get("Optimized_Draft"):
                            continue

                        title = row.get("Title", "")

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
                            all_warnings      = optimizer_result.get("warnings", [])
                            blocking_warnings = [
                                w for w in all_warnings
                                if not any(nb in w for nb in non_blocking_warnings)
                            ]

                            # "Ready to Publish" → Publisher picks it up
                            # "Needs Review"     → human must check first
                            new_status = "Ready to Publish" if not blocking_warnings else "Needs Review"

                            self._update_cells(plan_sheet, index, {
                                "Status":          new_status,
                                "Optimized_Draft": optimizer_result.get("html",            ""),
                                "Schema":          optimizer_result.get("schema",           ""),
                                "MetaTitle":       optimizer_result.get("meta_title",       ""),
                                "MetaDescription": optimizer_result.get("meta_description", ""),
                                "UrlSlug":         optimizer_result.get("url_slug",         ""),
                                "WordCount":       optimizer_result.get("word_count",        ""),
                                "Reviewer_Notes":  "; ".join(all_warnings)
                                                   or row.get("Reviewer_Notes", "")
                            })
                            self._log("info", "Optimizer", title, new_status,
                                      f"Slug: {optimizer_result.get('url_slug', 'N/A')}")
                            self.db.log_task("Optimizer", title, "SUCCESS",
                                             f"Status → {new_status}")

                            if test_mode:
                                print("🛑 Test Mode: stopping Optimizer after 1 run.")
                                time.sleep(10)
                                break

                        time.sleep(10)

                    except Exception as row_e:
                        title = row.get("Title", f"Row {index}")
                        self._log("error", "Optimizer", title, "ROW_EXCEPTION", str(row_e))
                        self.db.log_task("Optimizer", title, "ROW_EXCEPTION", str(row_e))
                        try:
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": f"Pipeline error: {str(row_e)[:200]}"
                            })
                        except Exception:
                            pass
                        continue

            except Exception as e:
                self._log("error", "Optimizer", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 6: PUBLISHER
        #
        # ✅ FIX vs original:
        #   - Visibility ya fue escrita en el Sheet por Agent 3+4
        #     → Publisher lee content_row["Visibility"] directamente
        #     → No recomputar force_hidden desde Reviewer_Notes aquí
        #   - pub_status == "Live" (publisher.py siempre retorna "Live")
        #   - Usa publish_result["is_hidden"] para decidir Telegram + Sheet labels
        #   - ChromaDB save siempre ocurre en Orchestrator (hidden o public)
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 6:
            print("\n" + "─" * 60)
            print("🚀 AGENT 6: PUBLISHER")

            telegram_on_hidden = self._get_rule(
                "publisher", "telegram_on_hidden", default=True
            )

            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    try:
                        if row.get("Status") != "Ready to Publish":
                            continue
                        if not row.get("Optimized_Draft"):
                            continue
                        if row.get("Published_Status"):
                            continue

                        title = row.get("Title", "")

                        # ✅ FIX: Visibility ya está en el Sheet desde Agent 3+4
                        # No recomputar — leer directo de la fila
                        visibility_from_sheet = str(row.get("Visibility", "")).strip().lower()
                        is_hidden_pre         = visibility_from_sheet == "hidden"

                        if is_hidden_pre:
                            print(f"   ⚠️  Visibility=hidden — publishing as Live/Hidden.")

                        optimizer_result = {
                            "status":           "Ready for Approval",
                            "html":             row.get("Optimized_Draft",   ""),
                            "schema":           row.get("Schema",            ""),
                            "meta_title":       row.get("MetaTitle",         ""),
                            "meta_description": row.get("MetaDescription",   ""),
                            "url_slug":         row.get("UrlSlug",           ""),
                            "word_count":       row.get("WordCount",          0),
                            "warnings":         [],
                        }

                        # content_row passed to Publisher already has Visibility
                        # from the Sheet — Publisher reads it directly
                        publish_result = publisher.publish_post(
                            content_row      = row,
                            optimizer_result = optimizer_result,
                            config           = self.config
                        )
                        pub_status = publish_result.get("status", "")

                        # ── Case 1: Upload FAILED ──────────────────────────────
                        if pub_status in ("BLOCKED", "FAILED"):
                            error_note = "; ".join(
                                publish_result.get("errors",   []) +
                                publish_result.get("warnings", [])
                            )
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": error_note
                            })
                            self._log("error", "Publisher", title, pub_status, error_note)
                            self.db.log_task("Publisher", title, pub_status, error_note)
                            print(f"   ❌ Upload failed — no Telegram sent.")

                        # ── Case 2: Upload SUCCEEDED ───────────────────────────
                        # ✅ FIX: pub_status == "Live" (publisher always returns "Live")
                        else:
                            admin_url    = publish_result.get("admin_url",  "")
                            is_hidden    = publish_result.get("is_hidden",  False)  # ✅ FIX
                            hidden_reason= publish_result.get("hidden_reason", "")
                            warning_note = "; ".join(publish_result.get("warnings", []))

                            # ✅ FIX: Sheet labels use is_hidden from Publisher result
                            sheet_status     = "Live (Hidden)" if is_hidden else "Live"
                            published_label  = "Hidden - Needs Review" if is_hidden else "Live"

                            self._update_cells(plan_sheet, index, {
                                "Status":           sheet_status,
                                "Published_Status": published_label,
                                "AdminURL":         admin_url,
                                "Reviewer_Notes":   warning_note or row.get("Reviewer_Notes", ""),
                                "Visibility":       "hidden" if is_hidden else "public",
                            })
                            self._log("info", "Publisher", title, sheet_status, admin_url)
                            self.db.log_task("Publisher", title, "SUCCESS",
                                             f"{sheet_status} → {admin_url}")

                            # ── ChromaDB: always save (hidden or live) ─────────
                            # Post exists in Shopify → Planner should not re-generate it
                            if self.chroma:
                                try:
                                    self.chroma.save_post({
                                        "title":          title,
                                        "url_slug":       row.get("UrlSlug",      ""),
                                        "keyword":        row.get("Keyword",       ""),
                                        "section":        row.get("Section",       ""),
                                        "summary":        row.get("Summary",       ""),
                                        "published_date": row.get("ScheduledDate", ""),
                                        "admin_url":      admin_url,
                                    })
                                    print("   🧠 Saved to ChromaDB memory.")
                                except Exception as chroma_e:
                                    print(f"   ⚠️  ChromaDB save failed (non-critical): {chroma_e}")

                            # ── Telegram ───────────────────────────────────────
                            if is_hidden and telegram_on_hidden:
                                self.send_telegram_alert(
                                    title     = title,
                                    admin_url = admin_url,
                                    hidden    = True,
                                    reason    = hidden_reason or row.get("Reviewer_Notes", "")
                                )
                            else:
                                self.send_telegram_alert(
                                    title     = title,
                                    admin_url = admin_url,
                                    hidden    = False
                                )

                            if test_mode:
                                print("🛑 Test Mode: stopping Publisher after 1 run.")
                                break

                    except Exception as row_e:
                        title = row.get("Title", f"Row {index}")
                        self._log("error", "Publisher", title, "ROW_EXCEPTION", str(row_e))
                        self.db.log_task("Publisher", title, "ROW_EXCEPTION", str(row_e))
                        try:
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": f"Pipeline error: {str(row_e)[:200]}"
                            })
                        except Exception:
                            pass
                        continue

            except Exception as e:
                self._log("error", "Publisher", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # PIPELINE COMPLETE
        # ══════════════════════════════════════════════════════════════════════
        print("\n✅ Pipeline run complete.")
        logging.info("Pipeline run complete.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = ContentOrchestrator(sheet_name="Blog_agent_ai")
    bot.run(start_from_agent=1)
