import os
import re
import json
import time
import logging
import requests
import gspread
from datetime import date
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
                    if not bt:
                        continue

                    def _bool(val):
                        if isinstance(val, bool): return val
                        return str(val).strip().upper() == "TRUE"

                    blog_types[bt] = {
                        "includes_faq":              _bool(row.get("includes_faq",              False)),
                        "target_product_link":        _bool(row.get("target_product_link",        False)),
                        "actionable_section_title":   str(row.get("actionable_section_title",    "")),
                        "actionable_section_desc":    str(row.get("actionable_section_desc",     "")),
                    }
            except Exception:
                blog_types = {}
                print("   ⚠️  Config_BlogTypes tab not found — defaults will be used.")

                        # ── Config_TopicMap ───────────────────────────────────────────────
            try:
                topic_map_rows = self.sh.worksheet("Config_TopicMap").get_all_records()
            except Exception:
                topic_map_rows = []
                print("   ⚠️  Config_TopicMap tab not found — blog_type balance disabled.")

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
                "topic_map":   topic_map_rows,   
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
            print(f"   TopicMap  : {len(topic_map_rows)} entries")
            print(f"   Rules     : {rule_count} rule(s) loaded")
            print(f"   Gemini    : {'✅' if os.getenv('GEMINI_API_KEY')    else '❌ MISSING'}")
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
    # ══════════════════════════════════════════════════════════════════════
        # DAILY SUMMARY
        # ══════════════════════════════════════════════════════════════════════
    def _send_daily_summary(self, plan_sheet):
            """Envía resumen diario por Telegram al terminar el run."""
            try:
                rows = plan_sheet.get_all_records()
                total      = len([r for r in rows if r.get("Title")])
                published  = len([r for r in rows if r.get("Published_Status") == "published"])
                pending    = len([r for r in rows if r.get("Status") == "Pending Approval"])
                needs_rev  = len([r for r in rows if r.get("Status") == "Needs Review"])
                ready      = len([r for r in rows if r.get("Status") == "Ready to Publish"])

                msg = (
                    f"📊 *Glomend Blog Agent — Daily Report*\n"
                    f"📅 {date.today().strftime('%B %d, %Y')}\n\n"
                    f"✅ Published : {published}\n"
                    f"🚀 Ready     : {ready}\n"
                    f"⏳ Peding   : {pending}\n"
                    f"⚠️  Needs Rev : {needs_rev}\n"
                    f"📝 Total     : {total}\n"
                )
                if needs_rev > 0:
                    stuck = [r.get("Title","")[:40] for r in rows
                            if r.get("Status") == "Needs Review"][:3]
                    msg += f"\n🔴 *Stuck posts:*\n" + "\n".join(f"- {t}..." for t in stuck)

                self._send_telegram(msg)
            except Exception as e:
                print(f"   ⚠️  Daily summary failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, start_from_agent: int = 1):
        if not self.check_system_status():
            return
        if not self.load_configurations():
            return
        try:
            plan_sheet_temp = self.sh.worksheet("Content_Plan")
            all_rows = plan_sheet_temp.get_all_records()
            for i, row in enumerate(all_rows, start=2):
                if (row.get("Status") == "Needs Review"
                        and "word count" in str(row.get("Reviewer_Notes", "")).lower()):
                    self._update_cells(plan_sheet_temp, i, {
                        "Status":        "Pending Approval",
                        "Draft_Content": "",
                        "Reviewer_Notes": "Auto-requeued: word count issue",
                    })
                    print(f"   🔄 Auto-requeued: '{row.get('Title', '')}'")
        except Exception as e:
            print(f"   ⚠️  Auto-reset failed: {e}")

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
                existing_records = plan_sheet.get_all_records()
                existing_titles  = [r.get("Title", "") for r in existing_records if r.get("Title")]

                # ── NEW: pass existing_records + occupied_dates to config ──────
                self.config["existing_records"] = existing_records
                self.config["occupied_dates"]   = [
                    r.get("ScheduledDate", "") for r in existing_records
                    if r.get("ScheduledDate")
                ]

                # ── NEW: pick hungriest section based on TopicMap gaps ─────────
                from planner import compute_coverage_gap

                def _pick_hungriest_section(all_sections):
                    coverage_gaps = compute_coverage_gap(
                        existing_records,
                        self.config.get("topic_map", []),
                        self.config
                    )
                    section_scores = {}
                    for g in coverage_gaps:
                        if g["priority"] != "SATURATED":
                            s = g["section"]
                            section_scores[s] = section_scores.get(s, 0) + g["remaining"]

                    if not section_scores:
                        print("   ℹ️  All sections SATURATED or no TopicMap — using first section.")
                        return all_sections[:sections_limit]

                    best_name = max(section_scores, key=section_scores.get)
                    print(f"   🏆 Hungriest section: '{best_name}' "
                          f"(total remaining: {section_scores[best_name]})")
                    matched = [s for s in all_sections if s["Name"] == best_name]
                    return matched if matched else all_sections[:sections_limit]

                sections_to_plan = _pick_hungriest_section(self.config["sections"])
                # ─────────────────────────────────────────────────────────────

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
                        # ── AUTO-APPROVE: cambia status a "Pending Research" para que
                        # el Researcher lo tome sin intervención humana ─────────────────
                        # (el Researcher ya filtra por "Pending Approval" — no hay que cambiar nada más)
                        self._log("info", "Planner", "batch", f"Added {len(approved_rows)} topic(s) — auto-approved")
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
                        if row.get("Status") not in ("Pending Approval", "Needs Review"):
                            continue
                        if not row.get("Research_Brief"):
                            continue
                        if row.get("Draft_Content") and row.get("Status") != "Needs Review":
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
                        final_status              = "hidden"
                        required_fixes_for_writer = []
                        word_count_min            = int(
                            self.config["brand"].get("default_word_count_min", 800)
                        )

                        # ── Helper: check word count ──────────────────────────
                        def _word_count_ok(html_text):
                            text = re.sub(r'<[^>]+>', ' ', html_text or "")
                            return len(text.split()) >= word_count_min

                        # ══════════════════════════════════════════════════════
                        # PHASE 1: Write with current topic (max_retries)
                        # ══════════════════════════════════════════════════════
                        for attempt in range(1, max_retries + 1):
                            print(f"\n   🔄 Attempt {attempt}/{max_retries}: Writing '{title}'...")

                            write_result = writer.write_draft(
                                content_row     = row,
                                research_result = research_result,
                                config          = self.config,
                                previous_draft  = best_draft or "",
                                required_fixes  = required_fixes_for_writer,
                            )

                            write_status = write_result.get("status", "")

                            if write_status in ("HARD_FAIL", "BLOCKED"):
                                print(f"   🚫 Writer {write_status} on attempt {attempt}.")
                                self._log("error", "Writer", title, write_status,
                                          str(write_result.get("errors") or write_result.get("violations")))
                                break

                            draft = write_result.get("html", "")
                            if draft:
                                best_draft = draft

                            # ── Word count check ──────────────────────────────
                            if not _word_count_ok(draft):
                                wc = len(re.sub(r'<[^>]+>', ' ', draft).split())
                                print(f"   ⚠️  Word count too low ({wc} words < {word_count_min}) "
                                      f"— requesting expansion (attempt {attempt}/{max_retries})...")
                                required_fixes_for_writer = [
                                    f"EXPAND CONTENT: current draft is only ~{wc} words. "
                                    f"Must reach at least {word_count_min} words. "
                                    f"Add more detail, examples, and actionable sections."
                                ]
                                continue  # retry with expansion instruction

                            # ── Reviewer ──────────────────────────────────────
                            print(f"   🛡️  Reviewer evaluating attempt {attempt}...")
                            review_result = reviewer.review_draft(
                                content_row     = row,
                                draft_text      = draft,
                                research_result = research_result,
                                config          = self.config,
                            )

                            review_status    = review_result.get("status",           "")
                            reviewer_summary = review_result.get("reviewer_summary", "")
                            required_fixes   = review_result.get("required_fixes",   [])
                            violations       = review_result.get("violations",        [])

                            print(f"      Reviewer: {review_status}")

                            if review_status in ("PASS", "PASS_WITH_NOTES"):
                                best_draft            = draft
                                best_reviewer_summary = reviewer_summary
                                final_status          = "public"
                                print(f"   ✅ GREEN FLAG on attempt {attempt} — queued for Live/Public.")
                                break

                            best_reviewer_summary     = reviewer_summary
                            required_fixes_for_writer = list(required_fixes)
                            for v in violations:
                                fix = f"COMPLIANCE VIOLATION: {v}"
                                if fix not in required_fixes_for_writer:
                                    required_fixes_for_writer.append(fix)

                            print(f"   🔁 {len(required_fixes_for_writer)} issue(s) — rewriting...")

                        # ══════════════════════════════════════════════════════
                        # PHASE 2: Word count still failing → swap topic
                        # ══════════════════════════════════════════════════════
                        if final_status != "public" and best_draft and not _word_count_ok(best_draft):
                            print(f"\n   🔀 Word count still failing after {max_retries} attempts "
                                  f"— requesting NEW TOPIC from Planner for section '{row.get('Section')}'...")

                            self._log("warning", "Writer", title, "TOPIC_SWAP",
                                      f"Word count < {word_count_min} after {max_retries} retries.")

                            try:
                                existing_records_fresh = plan_sheet.get_all_records()
                                existing_titles_fresh  = [
                                    r.get("Title", "") for r in existing_records_fresh
                                    if r.get("Title")
                                ]
                                self.config["existing_records"] = existing_records_fresh

                                retry_plan = planner.plan_next_posts(
                                    sections_config   = [s for s in self.config["sections"]
                                                         if s["Name"] == row.get("Section")],
                                    existing_titles   = existing_titles_fresh,
                                    config            = self.config,
                                    posts_per_section = 1,
                                )

                                new_topic_rows = retry_plan.get("new_rows", [])

                                if new_topic_rows:
                                    new_topic    = new_topic_rows[0]
                                    new_row      = dict(row)  # copy original row
                                    new_row.update({
                                        "Title":             new_topic.get("Title",             title),
                                        "Keyword":           new_topic.get("Keyword",           row.get("Keyword",  "")),
                                        "SecondaryKeywords": new_topic.get("SecondaryKeywords", ""),
                                        "Summary":           new_topic.get("Summary",           ""),
                                        "BlogType":          new_topic.get("BlogType",          row.get("BlogType", "")),
                                        "TopicCluster":      new_topic.get("TopicCluster",      ""),
                                    })
                                    new_title = new_row["Title"]
                                    print(f"   🆕 New topic: '{new_title}' — retrying Writer...")

                                    # Re-research the new topic
                                    new_research = researcher.research_topic(
                                        content_row    = new_row,
                                        valid_sections = valid_sections,
                                        source_policy  = source_policy,
                                    )
                                    if new_research.get("status") != "BLOCKED":
                                        new_research_json = json.dumps(new_research)

                                        # 3 fresh attempts with new topic
                                        best_draft_swap            = None
                                        required_fixes_for_writer  = []

                                        for swap_attempt in range(1, max_retries + 1):
                                            print(f"   🔄 Swap attempt {swap_attempt}/{max_retries}: "
                                                  f"Writing '{new_title}'...")

                                            swap_write = writer.write_draft(
                                                content_row     = new_row,
                                                research_result = new_research,
                                                config          = self.config,
                                                previous_draft  = best_draft_swap or "",
                                                required_fixes  = required_fixes_for_writer,
                                            )

                                            swap_draft = swap_write.get("html", "")
                                            if swap_draft:
                                                best_draft_swap = swap_draft

                                            if not _word_count_ok(swap_draft):
                                                wc = len(re.sub(r'<[^>]+>', ' ', swap_draft).split())
                                                required_fixes_for_writer = [
                                                    f"EXPAND CONTENT: only ~{wc} words. "
                                                    f"Must reach {word_count_min}+."
                                                ]
                                                continue

                                            # Reviewer on swap draft
                                            swap_review = reviewer.review_draft(
                                                content_row     = new_row,
                                                draft_text      = swap_draft,
                                                research_result = new_research,
                                                config          = self.config,
                                            )

                                            if swap_review.get("status") in ("PASS", "PASS_WITH_NOTES"):
                                                best_draft            = swap_draft
                                                best_reviewer_summary = swap_review.get("reviewer_summary", "")
                                                final_status          = "public"
                                                title                 = new_title
                                                row                   = new_row
                                                research_result       = new_research
                                                print(f"   ✅ New topic passed on swap attempt {swap_attempt}.")
                                                break

                                            required_fixes_for_writer = swap_review.get("required_fixes", [])
                                            best_reviewer_summary     = swap_review.get("reviewer_summary", "")

                                        if final_status != "public" and best_draft_swap:
                                            # Use best swap draft even if not perfect
                                            best_draft            = best_draft_swap
                                            best_reviewer_summary = (best_reviewer_summary or
                                                                      "Topic swapped — word count or review issues remain.")
                                            title = new_title
                                            row   = new_row
                                            print(f"   ⚠️  Swap topic also struggled — publishing best draft as hidden.")

                            except Exception as swap_e:
                                print(f"   ⚠️  Topic swap failed: {swap_e} — using best draft from original topic.")
                                self._log("warning", "Writer", title, "TOPIC_SWAP_FAILED", str(swap_e))

                        # ══════════════════════════════════════════════════════
                        # POST-LOOP: save whatever we have — never stop pipeline
                        # ══════════════════════════════════════════════════════
                        if not best_draft:
                            self._update_cells(plan_sheet, index, {
                                "Status":         "Needs Review",
                                "Reviewer_Notes": "All attempts failed (HARD_FAIL/BLOCKED) — manual intervention needed.",
                            })
                            self._log("error", "Writer", title, "NO_DRAFT_PRODUCED")
                            self.db.log_task("Writer", title, "NO_DRAFT_PRODUCED",
                                             "No valid draft after all retries.")
                            continue  # ← skip to next row, don't stop

                        self._update_cells(plan_sheet, index, {
                            "Draft_Content":  best_draft,
                            "WordCount":      str(len(re.sub(r'<[^>]+>', ' ', best_draft).split())),
                            "Reviewer_Notes": best_reviewer_summary,
                            "Status":         "Ready_To_Publish",
                            "Visibility":     "hidden" if final_status == "hidden" else "",
                            "Title":          row.get("Title", title),  # update if topic was swapped
                        })

                        status_label = (
                            "APPROVED → queued for Live/Public"
                            if final_status == "public"
                            else f"Max retries → queued as Live/Hidden (word count or review issues)"
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
                        continue  # ← siempre continúa

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
                default="Internal link inventory unavailable,schema_type missing, \
                        No inline images,Featured image is attached,Word count"
            )
            non_blocking_warnings = [
                w.strip() for w in str(non_blocking_raw).split(",") if w.strip()
            ]

            try:
                existing_records = plan_sheet.get_all_records()

                for index, row in enumerate(existing_records, start=2):
                    try:
                        # ✅ FIX: "Ready_To_Publish" is what Agent 3+4 writes
                        OPTIMIZER_ELIGIBLE = ("Ready_To_Publish", "Needs Review")
                        if row.get("Status") not in OPTIMIZER_ELIGIBLE:
                            continue
                        if not row.get("Draft_Content"):
                            continue
                        # Re-evalúa "Needs Review" aunque ya tenga Optimized_Draft
                        if row.get("Optimized_Draft") and row.get("Status") != "Needs Review":
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
            published_today = 0
            max_per_run = int(self._get_rule("publisher", "max_per_run", default=1) or 1)

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

                        # ── NEW: only publish if ScheduledDate is today or past ──
                        from datetime import date
                        today_str      = date.today().strftime("%Y-%m-%d")
                        scheduled_date = str(row.get("ScheduledDate", "")).strip()
                        if scheduled_date and scheduled_date > today_str:
                            print(f"   ⏳ '{row.get('Title', '')}' scheduled for "
                                  f"{scheduled_date} — skipping today.")
                            continue
                        if published_today >= max_per_run:
                            print(f"   ⏸️  Max {max_per_run} post(s) per run reached — stopping.")
                            break

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

                            published_today += 1

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
        self._send_daily_summary(plan_sheet)
        print("\n✅ Pipeline run complete.")
        logging.info("Pipeline run complete.")

        

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sheet_name = os.getenv("SHEET_NAME", "Blog_agent_ai")
    bot = ContentOrchestrator(sheet_name=sheet_name)
    bot.run(start_from_agent=1)