"""
utils.py — Shared utility functions used across all agents.

Think of this as a shared toolbox. Instead of repeating the same logic
(connecting to Sheets, parsing JSON, sending Telegram messages) in every
agent file, we write each function once here and import it wherever needed.

Functions:
- get_sheet()             → connects to Google Sheets using service_account.json
- load_agent_rules()      → reads the Agent_Rules tab into a nested dictionary
- load_prompts()          → reads the Prompts tab and fills in {brand} variables
- safe_json_parse()       → safely converts a JSON string to a Python dict
- send_telegram_alert()   → sends a Telegram notification when a draft is ready
"""

import json
import logging
import os
import requests
import gspread
from dotenv import load_dotenv

# Load all variables from the .env file into the environment
load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS CONNECTION
# ──────────────────────────────────────────────────────────────────────────────

def get_sheet(sheet_name: str = None):
    """
    Opens and returns a Google Sheets spreadsheet object.

    What it does:
    - If no sheet_name is provided, reads SHEET_NAME from the .env file.
    - Authenticates using the service_account.json credentials file.
    - Returns a gspread Spreadsheet object you can call .worksheet() on.

    Why it exists:
    - Every agent needs to connect to the Sheet. This avoids repeating
      the same 3 lines of connection code in every file.

    Example:
        sh = get_sheet()
        ws = sh.worksheet("Content_Plan")
    """
    if not sheet_name:
        # Read from .env so you can change the sheet name without editing code
        sheet_name = os.getenv("SHEET_NAME", "Blog_agent_ai")

    gc = gspread.service_account(filename="service_account.json")
    return gc.open(sheet_name)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT RULES LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_agent_rules(sh) -> dict:
    """
    Reads the 'Agent_Rules' tab and returns a nested dictionary of rules.

    What it does:
    - Loads all rows from the Agent_Rules tab in Google Sheets.
    - Groups them by agent name into a nested dictionary.

    Returns a structure like:
    {
        "orchestrator": {"max_retries": "3", "posts_per_section": "1"},
        "writer":       {"word_count_min": "1000", "tone": "warm, empowering"},
        "reviewer":     {"forbidden_words": "cures, treats, heals"}
    }

    Why it exists:
    - Agent behavior (retries, word counts, tone) should live in the Sheet,
      not buried in code. This makes it editable per client without touching Python.

    If the tab doesn't exist, returns an empty dict so the pipeline can still
    run using the default values defined in each agent.
    """
    try:
        ws   = sh.worksheet("Agent_Rules")
        rows = ws.get_all_records()  # Each row → {agent_name, rule_key, rule_value}

        rules = {}
        for row in rows:
            agent = row.get("agent_name", "").strip()
            key   = row.get("rule_key",   "").strip()
            value = row.get("rule_value", "")

            if not agent or not key:
                continue  # Skip completely blank rows

            if agent not in rules:
                rules[agent] = {}
            rules[agent][key] = value

        return rules

    except Exception as e:
        logging.warning(f"[Utils] Could not load Agent_Rules tab: {e}")
        return {}  # Return empty dict — agents will use their default values


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_prompts(sh, config: dict) -> dict:
    """
    Reads the 'Prompts' tab and fills in dynamic {variable} placeholders.

    What it does:
    - Loads all rows from the Prompts tab.
    - Replaces {variable} placeholders with real values from Config_Brand.
      Example: "You are a writer for {brand_name}" → "You are a writer for Glomend"

    Returns a structure like:
    {
        "writer":    {"system_prompt": "You are a wellness writer for Glomend..."},
        "reviewer":  {"system_prompt": "Check FDA compliance. Avoid: cures..."},
        "publisher": {"telegram_message": "🚀 New post: Glomend..."}
    }

    Why it exists:
    - The instructions sent to Gemini (called 'prompts') should be editable
      from the Sheet without changing code. This lets you customize tone, rules,
      or language for any client just by editing a cell.

    If a {variable} in the template doesn't exist in config, it's left as-is
    rather than crashing. If the tab doesn't exist, returns an empty dict.
    """
    try:
        ws   = sh.worksheet("Prompts")
        rows = ws.get_all_records()  # Each row → {agent_name, prompt_key, prompt_text}

        prompts = {}
        for row in rows:
            agent = row.get("agent_name", "").strip()
            key   = row.get("prompt_key", "").strip()
            text  = row.get("prompt_text", "")

            if not agent or not key:
                continue

            # Replace {brand_name}, {fda_disclaimer_text}, etc. with real values
            try:
                text = text.format(**config)
            except KeyError:
                pass  # If a variable doesn't exist in config, leave the placeholder as-is

            if agent not in prompts:
                prompts[agent] = {}
            prompts[agent][key] = text

        return prompts

    except Exception as e:
        logging.warning(f"[Utils] Could not load Prompts tab: {e}")
        return {}  # Return empty dict — agents will use their built-in prompts


# ──────────────────────────────────────────────────────────────────────────────
# JSON PARSER
# ──────────────────────────────────────────────────────────────────────────────

def safe_json_parse(raw_value, fallback=None):
    """
    Safely converts a JSON string into a Python dictionary (or list).

    What it does:
    - Takes a string (usually from a Google Sheet cell like Research_Brief).
    - Tries to parse it as JSON.
    - If it fails for ANY reason, returns the fallback value instead of crashing.

    Why it exists:
    - Sheet cells that store JSON can sometimes be empty, malformed, or not
      a string at all. This prevents the pipeline from crashing on bad data.

    Examples:
        safe_json_parse('{"status": "PASS"}')  → {"status": "PASS"}
        safe_json_parse("")                     → {}
        safe_json_parse("not json")             → {}
        safe_json_parse(None, fallback=[])      → []
    """
    if not raw_value or not isinstance(raw_value, str):
        return fallback if fallback is not None else {}
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return fallback if fallback is not None else {}


# ──────────────────────────────────────────────────────────────────────────────
# TELEGRAM NOTIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def send_telegram_alert(
    title:            str,
    admin_url:        str = "",
    token:            str = "",
    chat_id:          str = "",
    message_template: str = None
):
    """
    Sends a Telegram message when a new blog draft is ready for review.

    What it does:
    - Uses the bot token and chat ID to send a formatted message.
    - If a custom template exists in the Prompts tab, uses that instead of
      the default format. Use {title} and {admin_url} as placeholders.

    Parameters:
    - title:            The blog post title to show in the notification.
    - admin_url:        The Shopify admin URL for reviewing the draft.
    - token:            Telegram Bot Token (always comes from .env — sensitive).
    - chat_id:          Telegram Chat ID (can come from Config_System Sheet).
    - message_template: Optional custom message from the Prompts tab.

    Security note:
    - The BOT TOKEN is a password and must stay in .env only.
    - The CHAT ID is not sensitive and can live in the Sheet per client.

    If credentials are missing, skips silently without crashing the pipeline.
    """
    if not token or not chat_id:
        print("   ⚠️ Telegram credentials missing — skipping notification.")
        return

    # Use custom template from Prompts tab if provided, otherwise use default
    if message_template:
        message = message_template.format(
            title     = title,
            admin_url = admin_url or "Check Shopify Admin"
        )
    else:
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

def safe_agent_run(agent_name: str, db=None):
    """
    Decorator that wraps any agent function in a fail-safe try/except.
    If the agent crashes completely, logs the error and returns a safe
    default result instead of crashing the entire pipeline.

    Usage:
        @safe_agent_run("Researcher")
        def run_researcher():
            ...
    """
    import functools
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"[{agent_name}] CRITICAL EXCEPTION: {e}")
                print(f"🚨 [{agent_name}] crashed but pipeline continues: {e}")
                if db:
                    db.log_task(agent_name, "N/A", "CRITICAL_EXCEPTION", str(e))
                return {"status": "EXCEPTION", "errors": [str(e)]}
        return wrapper
    return decorator