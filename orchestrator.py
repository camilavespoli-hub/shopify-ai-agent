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

# ChromaManager — graceful fallback if file not yet in place
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
}


class ContentOrchestrator:

    def __init__(self, sheet_name="Blog_agent_ai"):
        print("🤖 Initializing Orchestrator...")

        # ── AWS detection ─────────────────────────────────────────────────────
        # Locally  → files live in the project folder (DATA_DIR = ".")
        # On AWS   → files live in /mnt/efs/ (persistent EFS volume)
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

        # SQLite — path adjusts for local vs AWS
        self.db = DatabaseManager(
            db_path=os.path.join(self.DATA_DIR, "pipeline.db")
        )

        # ChromaDB — initialized after load_configurations() (needs brand_name)
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
        """
        Reads a rule from Agent_Rules tab (self.config['agent_rules']).
        Auto-converts TRUE/FALSE strings → bool, numeric strings → int/float.
        Returns default if the key is not found.
        """
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
        """
        Reads a setting from Config_System (self.config['system']).
        Auto-converts TRUE/FALSE and numeric strings.
        """
        val = self.config.get("system", {}).get(key, default)
        if isinstance(val, str):
            if val.upper() == "TRUE":  return True
            if val.upper() == "FALSE": return False
            try:    return int(val)
            except (ValueError, TypeError): pass
        return val

    def _planner_row_to_sheet_list(self, row_dict):
        """
        Converts a Planner output dict to a 21-column list matching COL.
        Columns A–U exactly. All agent-filled columns start empty.

        FIX vs original: TrendScore was incorrectly placed at col J
        (Reviewer_Notes) and col K (Research_Brief). Now both are empty
        at planning time — only Status/Title/Section/etc. are filled.
        """
        return [
            "",                                            # A  RowID (auto)
            row_dict.get("Status",            "Pending Approval"),  # B
            row_dict.get("Title",             ""),         # C
            row_dict.get("Section",           ""),         # D
            row_dict.get("Keyword",           ""),         # E
            row_dict.get("SecondaryKeywords", ""),         # F
            row_dict.get("Summary",           ""),         # G
            row_dict.get("ScheduledDate",     ""),         # H
            row_dict.get("ScheduledTime",     "09:00 AM ET"),  # I
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
            # ── Config_Brand ──────────────────────────────────────────────────
            brand_rows = self.sh.worksheet("Config_Brand").get_all_records()
            brand_flat = {
                row["Field Name"]: row["Value"]
                for row in brand_rows
                if row.get("Field Name")
            }

            # ── Config_System ─────────────────────────────────────────────────
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

            # ── Agent_Rules ───────────────────────────────────────────────────
            # Stored as nested dict: { agent_name: { rule_key: rule_value } }
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

            # ── Prompts (optional) ────────────────────────────────────────────
            # Stored as nested dict: { agent_name: { prompt_key: prompt_text } }
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

            # ── Config_Sections ───────────────────────────────────────────────
            sections = self.sh.worksheet("Config_Sections").get_all_records()

            # ── Config_Products (optional) ────────────────────────────────────
            try:
                product_rows = self.sh.worksheet("Config_Products").get_all_records()
            except Exception:
                product_rows = []
                print("   ⚠️  Config_Products tab not found.")

            # ── Config_Cadence (optional) ─────────────────────────────────────
            try:
                cadence_rows   = self.sh.worksheet("Config_Cadence").get_all_records()
                active_cadence = next(
                    (r for r in cadence_rows if str(r.get("Active", "")).upper() == "Y"), {}
                )
            except Exception:
                active_cadence = {}
                print("   ⚠️  Config_Cadence tab not found.")

            # ── Config_Planner (optional) ─────────────────────────────────────
            try:
                planner_rows = self.sh.worksheet("Config_Planner").get_all_records()
                planner_cfg  = {
                    row["Field_Name"]: row["Value"]
                    for row in planner_rows if row.get("Field_Name")
                }
            except Exception:
                planner_cfg = {}

            # ── Assemble full config ──────────────────────────────────────────
            self.config = {
                "brand":       brand_flat,
                "system":      system_flat,
                "agent_rules": agent_rules,
                "prompts":     prompts,
                "planner":     planner_cfg,
                "sections":    sections,
                "products":    product_rows,
                "cadence":     active_cadence,
            }

            # ── Required fields validation ────────────────────────────────────
            # disclaimer_text replaces fda_disclaimer_text (industry-agnostic)
            required = [
                "brand_voice_summary",
                "disclaimer_text",         # renamed from fda_disclaimer_text
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

            # ── Init ChromaDB now that we have brand_name ─────────────────────
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

            # ── Startup summary ───────────────────────────────────────────────
            rule_count = sum(len(v) for v in agent_rules.values())
            print(f"   Brand     : {brand_flat.get('brand_name',       '⚠️ not set')}")
            print(f"   Language  : {brand_flat.get('content_language', '⚠️ not set')}  "
                  f"| Industry: {brand_flat.get('industry', '⚠️ not set')}")
            print(f"   Sections  : {len(sections)}")
            print(f"   Products  : {len(product_rows)}")
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
        """Low-level sender — accepts a pre-formatted Markdown string."""
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
        """
        Sends a post-publish Telegram notification.

        hidden=False → 🚀 standard success message
        hidden=True  → ⚠️ warning — post is in Shopify but NOT visible

        Templates are read from the Prompts tab:
          publisher → telegram_message         (live posts)
          publisher → telegram_hidden_message  (hidden posts)
        Falls back to built-in defaults when templates are not defined.
        """
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

        # ── Runtime settings (Config_System + Agent_Rules) ────────────────────
        # start_from_agent argument wins over Config_System value
        system_start = self._get_system_setting("Start_From_Agent", default=1)
        if start_from_agent == 1 and system_start != 1:
            start_from_agent = int(system_start)

        test_mode = self._get_system_setting("Test_Mode", default=False)

        # max_retries: Agent_Rules (orchestrator) > Config_System > hardcoded 3
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
        # Proposes new blog topics from trend signals + section config.
        # If a proposed topic is semantically similar to an existing post
        # (ChromaDB), it retries silently up to max_topic_attempts times.
        # Pipeline NEVER stops due to duplicate detection.
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
                            # ChromaDB duplicate check (skipped if chroma not available)
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
                                is_dup = False  # accept all when chroma is disabled

                            if not is_dup:
                                approved_rows.append(proposed)
                                accepted = True
                                break

                            if duplicate_action == "warn":
                                print(f"   ⚠️  '{title}' is {score:.0%} similar to '{matched}' — allowed (warn mode).")
                                approved_rows.append(proposed)
                                accepted = True
                                break

                            # skip mode — ask Planner for a replacement topic
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
                            # All attempts exhausted — log and skip, pipeline continues
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
                    try:  # ← per-row isolation: one bad row never stops the agent
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
                        continue  # ← go to next row

            except Exception as e:
                self._log("error", "Researcher", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 3 + 4: WRITER → REVIEWER LOOP
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 3:
            print("\n" + "─" * 60)
            print("✍️  AGENT 3: WRITER  +  🛡️ AGENT 4: REVIEWER (auto-retry loop)")
            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    try:  # ← per-row isolation
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
                            print(f"   ⚠️  Row {index}: could not parse Research_Brief — skipping.")
                            continue

                        research_status = research_result.get("status", "")
                        if "BLOCKED" in research_status or "insufficient" in research_status:
                            print(f"   ⚠️  Row {index}: bad research status — skipping.")
                            continue

                        title          = row.get("Title", "")
                        previous_draft = ""
                        required_fixes = []
                        final_writer   = None
                        final_review   = None

                                                # ── Writer → Reviewer loop ────────────────────────────
                        for attempt in range(1, max_retries + 1):
                            print(f"\n   🔄 Attempt {attempt}/{max_retries}: Writing...")

                            # Ask the Writer agent to produce an HTML draft.
                            # On retries, it receives the previous draft + a list
                            # of specific fixes the Reviewer flagged.
                            writer_result = writer.write_draft(
                                content_row    = row,
                                research_result = research_result,
                                config          = self.config,
                                previous_draft  = previous_draft,
                                required_fixes  = required_fixes
                            )

                            # BLOCKED = a hard error (e.g. Gemini API failure).
                            # We stop retrying this article but continue to the next row.
                            if writer_result["status"] == "BLOCKED":
                                self._log("error", "Writer", title, "BLOCKED",
                                          str(writer_result.get("errors")))
                                self.db.log_task("Writer", title, "BLOCKED",
                                                 str(writer_result.get("errors")))
                                break

                            # HARD_FAIL = the draft contains serious violations
                            # (e.g. FDA-banned phrases) that can't be auto-fixed.
                            if writer_result["status"] == "HARD_FAIL":
                                self._log("warning", "Writer", title, "HARD_FAIL",
                                          str(writer_result.get("violations")))
                                self.db.log_task("Writer", title, "HARD_FAIL",
                                                 str(writer_result.get("violations")))
                                break

                            current_draft = writer_result.get("html", "")
                            print(f"   ✍️  Draft written "
                                  f"({writer_result.get('word_count', 0)} words). Reviewing...")

                            # Ask the Reviewer agent to check compliance, quality,
                            # and brand voice. It returns PASS, PASS_WITH_NOTES, or FAIL.
                            review_result = reviewer.review_draft(
                                content_row     = row,
                                draft_text      = current_draft,
                                research_result = research_result,
                                config          = self.config
                            )

                            review_status = review_result.get("status", "")
                            print(f"   🛡️  Review: {review_status}")

                            if review_status in ("PASS", "PASS_WITH_NOTES"):
                                # ✅ Draft passed — save final results and exit loop
                                final_writer = writer_result
                                final_review = review_result
                                print(f"   ✅ Passed on attempt {attempt}!")
                                break

                            # Draft failed — collect the fixes and send back to Writer
                            required_fixes = review_result.get("required_fixes", [])
                            previous_draft = current_draft
                            print(f"   ↩️  {len(required_fixes)} fix(es) sent back to Writer:")
                            for fix in required_fixes:
                                print(f"      - {fix}")

                            # If this was the last allowed attempt, save whatever
                            # we have and flag for manual review — don't discard the work
                            if attempt == max_retries:
                                final_writer = writer_result
                                final_review = review_result
                                print("   ⚠️  Max retries reached — saving draft as 'Needs Review'.")

                        # ── Save Writer + Reviewer results to the Sheet ────────
                        if final_writer and final_review:
                            review_status = final_review.get("status", "FAIL")
                            summary_note  = final_review.get("reviewer_summary", "")

                            # "Content Approved" → moves forward to Optimizer
                            # "Needs Review"     → waits for manual inspection
                            new_status = (
                                "Content Approved"
                                if review_status in ("PASS", "PASS_WITH_NOTES")
                                else "Needs Review"
                            )

                            self._update_cells(plan_sheet, index, {
                                "Draft_Content":  final_writer.get("html",       ""),
                                "WordCount":      final_writer.get("word_count",  0),
                                "Status":         new_status,
                                "Reviewer_Notes": summary_note
                            })
                            self._log("info", "Writer+Reviewer", title, new_status,
                                      f"Words: {final_writer.get('word_count', 0)}")
                            self.db.log_task(
                                "Writer+Reviewer", title,
                                "SUCCESS" if new_status == "Content Approved" else "NEEDS_REVIEW",
                                summary_note
                            )

                            if test_mode:
                                print("🛑 Test Mode: stopping after 1 article.")
                                time.sleep(10)
                                break

                        time.sleep(10)

                    except Exception as row_e:
                        # One row crashed — log it, mark it in the Sheet,
                        # and continue to the next row without stopping the pipeline
                        title = row.get("Title", f"Row {index}")
                        self._log("error", "Writer+Reviewer", title, "ROW_EXCEPTION", str(row_e))
                        self.db.log_task("Writer+Reviewer", title, "ROW_EXCEPTION", str(row_e))
                        try:
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": f"Pipeline error: {str(row_e)[:200]}"
                            })
                        except Exception:
                            pass
                        continue  # ← skip to next row

            except Exception as e:
                self._log("error", "Writer+Reviewer", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # AGENT 5: OPTIMIZER
        # Takes the approved draft and:
        #   - Finalizes the HTML with brand styles applied
        #   - Generates meta title, meta description, and URL slug
        #   - Builds schema markup (structured data for Google)
        #   - Checks word count limits from Config_Brand
        #
        # Word count: if the draft EXCEEDS the max, the Optimizer rewrites it
        # to fit — it does NOT throw an error. Pipeline always continues.
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 5:
            print("\n" + "─" * 60)
            print("⚙️  AGENT 5: OPTIMIZER")

            # Load non-blocking warning types from Agent_Rules.
            # These warnings are logged but will NOT stop the post from publishing.
            # Example: "Internal link inventory unavailable" — not critical.
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
                    try:  # ← per-row isolation: one crash doesn't stop all posts
                        if row.get("Status") != "Content Approved":
                            continue
                        if not row.get("Draft_Content"):
                            continue
                        # Skip if already optimized (prevents re-running on same row)
                        if row.get("Optimized_Draft"):
                            continue

                        title = row.get("Title", "")

                        # Build a minimal reviewer_result dict so the Optimizer
                        # knows the review passed (it uses this for context)
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
                            # A hard failure (e.g. Gemini error) — log and skip this row
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": "; ".join(optimizer_result.get("errors", []))
                            })
                            self._log("error", "Optimizer", title, "BLOCKED",
                                      str(optimizer_result.get("errors")))
                            self.db.log_task("Optimizer", title, "BLOCKED",
                                             str(optimizer_result.get("errors")))

                        else:
                            # Separate blocking warnings (need human review)
                            # from non-blocking ones (just informational)
                            all_warnings      = optimizer_result.get("warnings", [])
                            blocking_warnings = [
                                w for w in all_warnings
                                if not any(nb in w for nb in non_blocking_warnings)
                            ]

                            # "Ready to Publish" → Publisher can pick it up next
                            # "Needs Review"     → a human needs to check it first
                            new_status = "Ready to Publish" if not blocking_warnings else "Needs Review"

                            self._update_cells(plan_sheet, index, {
                                "Status":          new_status,
                                "Optimized_Draft": optimizer_result.get("html",             ""),
                                "Schema":          optimizer_result.get("schema",            ""),
                                "MetaTitle":       optimizer_result.get("meta_title",        ""),
                                "MetaDescription": optimizer_result.get("meta_description",  ""),
                                "UrlSlug":         optimizer_result.get("url_slug",          ""),
                                "WordCount":       optimizer_result.get("word_count",        ""),
                                # Keep any pre-existing notes if Optimizer has none
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
                        # Row-level crash — log, mark in Sheet, continue to next row
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
        # FAIL-SAFE DESIGN — this agent ALWAYS tries to publish something:
        #
        #   ✅ No red flags  →  published LIVE    →  standard Telegram notification
        #   ⚠️ Red flags     →  published HIDDEN  →  Telegram ⚠️ alert with reason
        #   ❌ Shopify error →  nothing uploaded  →  error logged, NO Telegram sent
        #                                             (there's nothing to review)
        #
        # "Red flags" = the Reviewer_Notes column has content AND the reviewer
        # rule "publish_hidden_on_warning" is TRUE in Agent_Rules.
        #
        # Telegram is sent ONLY when the post is confirmed live in Shopify.
        # ChromaDB always saves after a successful upload — hidden or live —
        # so the Planner won't generate duplicate topics in the future.
        # ══════════════════════════════════════════════════════════════════════
        if start_from_agent <= 6:
            print("\n" + "─" * 60)
            print("🚀 AGENT 6: PUBLISHER")

            # Read publisher-specific rules from Agent_Rules tab
            # publish_hidden_on_warning: if TRUE, posts with red flags go hidden
            publish_hidden_on_warning = self._get_rule(
                "publisher", "publish_hidden_on_warning", default=True
            )
            # telegram_on_hidden: if TRUE, sends a Telegram alert for hidden posts
            telegram_on_hidden = self._get_rule(
                "publisher", "telegram_on_hidden", default=True
            )

            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    try:  # ← per-row isolation
                        if row.get("Status") != "Ready to Publish":
                            continue
                        if not row.get("Optimized_Draft"):
                            continue
                        # Skip rows already published (prevents re-uploading)
                        if row.get("Published_Status"):
                            continue

                        title = row.get("Title", "")

                        # Determine if this post has red flags.
                        # A red flag = Reviewer_Notes is not empty after optimization.
                        has_red_flags = bool(row.get("Reviewer_Notes", "").strip())

                        # Decide visibility BEFORE calling the Publisher agent.
                        # This gives the Publisher agent explicit instructions.
                        force_hidden = has_red_flags and publish_hidden_on_warning
                        visibility   = "hidden" if force_hidden else "live"

                        if force_hidden:
                            print(f"   ⚠️  Red flags found — will publish as HIDDEN.")
                            print(f"   Reason: {row.get('Reviewer_Notes', '')[:100]}")

                        # Build the optimizer_result dict the Publisher expects.
                        # We read the data from the Sheet row (already saved by Optimizer).
                        optimizer_result = {
                            "status":           "Ready for Approval",
                            "html":             row.get("Optimized_Draft",   ""),
                            "schema":           row.get("Schema",            ""),
                            "meta_title":       row.get("MetaTitle",         ""),
                            "meta_description": row.get("MetaDescription",   ""),
                            "url_slug":         row.get("UrlSlug",           ""),
                            "word_count":       row.get("WordCount",          0),
                            "warnings":         [],
                            # Pass the visibility decision so Publisher
                            # sets published=false in Shopify when needed
                            "visibility":       visibility
                        }

                        publish_result = publisher.publish_post(
                            content_row      = row,
                            optimizer_result = optimizer_result,
                            config           = self.config
                        )
                        pub_status = publish_result.get("status", "")

                        # ── Case 1: Upload FAILED (never reached Shopify) ──────
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
                            # ⚠️ NO Telegram — the post is not in Shopify,
                            # there's nothing to review there
                            print(f"   ❌ Upload failed — no Telegram sent.")

                        # ── Case 2: Upload SUCCEEDED ───────────────────────────
                        else:
                            admin_url    = publish_result.get("admin_url", "")
                            warning_note = "; ".join(publish_result.get("warnings", []))

                            # "Live (Hidden)" → in Shopify but not visible to visitors
                            # "Live"          → fully visible on the blog
                            final_status    = "Live (Hidden)" if force_hidden else "Live"
                            published_label = "Hidden - Needs Review" if force_hidden else "Draft Created"

                            self._update_cells(plan_sheet, index, {
                                "Status":           final_status,
                                "Published_Status": published_label,
                                "AdminURL":         admin_url,
                                # Keep the original red flag note if no new warnings
                                "Reviewer_Notes":   warning_note or row.get("Reviewer_Notes", ""),
                                # Log which visibility was used for audit trail
                                "Visibility":       visibility
                            })
                            self._log("info", "Publisher", title, final_status, admin_url)
                            self.db.log_task("Publisher", title, "SUCCESS",
                                             f"{final_status} → {admin_url}")

                            # ── Save to ChromaDB memory ────────────────────────
                            # Always save — hidden OR live. Reason: the post exists
                            # in Shopify now, so the Planner should not write about
                            # the same topic again, regardless of visibility.
                            if self.chroma:
                                try:
                                    self.chroma.save_post({
                                        "title":          title,
                                        "url_slug":       row.get("UrlSlug",       ""),
                                        "keyword":        row.get("Keyword",        ""),
                                        "section":        row.get("Section",        ""),
                                        "summary":        row.get("Summary",        ""),
                                        "published_date": row.get("ScheduledDate",  ""),
                                        "admin_url":      admin_url,
                                    })
                                    print("   🧠 Saved to ChromaDB memory.")
                                except Exception as chroma_e:
                                    # ChromaDB failure never blocks publishing
                                    print(f"   ⚠️  ChromaDB save failed (non-critical): {chroma_e}")

                            # ── Send Telegram notification ─────────────────────
                            # Hidden post → alert with reason (so you can review it)
                            # Live post   → standard success notification
                            if force_hidden and telegram_on_hidden:
                                self.send_telegram_alert(
                                    title     = title,
                                    admin_url = admin_url,
                                    hidden    = True,
                                    reason    = row.get("Reviewer_Notes", "Red flags detected")
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
                        # Row-level crash — log, mark in Sheet, continue to next row.
                        # Even if one post's upload crashes, others still get published.
                        title = row.get("Title", f"Row {index}")
                        self._log("error", "Publisher", title, "ROW_EXCEPTION", str(row_e))
                        self.db.log_task("Publisher", title, "ROW_EXCEPTION", str(row_e))
                        try:
                            self._update_cells(plan_sheet, index, {
                                "Reviewer_Notes": f"Pipeline error: {str(row_e)[:200]}"
                            })
                        except Exception:
                            pass
                        continue  # ← keep publishing remaining posts

            except Exception as e:
                self._log("error", "Publisher", "N/A", "EXCEPTION", str(e))

        # ══════════════════════════════════════════════════════════════════════
        # PIPELINE COMPLETE
        # ══════════════════════════════════════════════════════════════════════
        print("\n✅ Pipeline run complete.")
        logging.info("Pipeline run complete.")


# ── Entry point ────────────────────────────────────────────────────────────────
# This block runs only when you execute the file directly:
#   python orchestrator.py
# It does NOT run when another file imports this module.
if __name__ == "__main__":
    bot = ContentOrchestrator(sheet_name="Blog_agent_ai")
    # start_from_agent=1 → runs all agents (1 through 6)
    # start_from_agent=5 → skips to Optimizer (useful for testing)
    # start_from_agent=6 → skips directly to Publisher
    bot.run(start_from_agent=1)
