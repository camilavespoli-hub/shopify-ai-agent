"""
Standalone publisher: reads Content_Plan sheet and publishes all
"Ready to Publish" articles without initializing the other agents.
"""
import os
import requests
import gspread
from dotenv import load_dotenv
from database_manager import DatabaseManager

load_dotenv()

try:
    from chroma_manager import ChromaManager
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from publisher import PublisherAgent

# ── Column map (matches orchestrator.py COL dict) ─────────────────────────────
COL = {
    "Status":           2,
    "Title":            3,
    "Section":          4,
    "Keyword":          5,
    "SecondaryKeywords":6,
    "Summary":          7,
    "ScheduledDate":    8,
    "Reviewer_Notes":   10,
    "Optimized_Draft":  14,
    "Schema":           15,
    "MetaTitle":        16,
    "MetaDescription":  17,
    "UrlSlug":          18,
    "AdminURL":         19,
    "Published_Status": 20,
    "Visibility":       21,
}

READY_STATUSES = ("Ready to Publish",)

def update_cells(worksheet, row_index, updates):
    cell_list = []
    for col_key, value in updates.items():
        col_num = COL.get(col_key)
        if not col_num:
            continue
        cell = worksheet.cell(row_index, col_num)
        cell.value = str(value) if value is not None else ""
        cell_list.append(cell)
    if cell_list:
        worksheet.update_cells(cell_list)

def send_telegram(title, admin_url, hidden=False, reason=""):
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    if hidden:
        message = (
            f"⚠️ *Article Published (Needs Review)*\n\n"
            f"*Title:* {title}\n"
            f"*Reason:* {reason or 'Red flags detected'}\n"
            f"*Review:* {admin_url or 'Check Shopify Admin'}"
        )
    else:
        message = (
            f"🚀 *New Blog Post Published!*\n\n"
            f"*Title:* {title}\n"
            f"*Review:* {admin_url or 'Check Shopify Admin'}"
        )
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        print("   📲 Telegram notification sent.")
    except Exception as e:
        print(f"   ⚠️  Telegram failed: {e}")


def main():
    sheet_name = os.getenv("SHEET_NAME", "Blog_agent_ai")
    print(f"📋 Connecting to Google Sheet: '{sheet_name}'...")
    client = gspread.service_account(filename="service_account.json")
    sh = client.open(sheet_name)
    plan_sheet = sh.worksheet("Content_Plan")
    print("✅ Connected.")

    db = DatabaseManager(db_path="./pipeline.db")

    chroma = None
    if CHROMA_AVAILABLE:
        try:
            brand_rows = sh.worksheet("Config_Brand").get_all_records()
            brand_name = next(
                (r["Value"] for r in brand_rows if r.get("Field Name") == "brand_name"),
                "default"
            )
            chroma = ChromaManager(persist_directory="./chroma_db", brand_name=brand_name)
            print(f"🧠 ChromaDB ready.")
        except Exception as e:
            print(f"⚠️  ChromaDB init failed: {e}")

    config = {"brand": {}, "system": {}}
    try:
        brand_rows = sh.worksheet("Config_Brand").get_all_records()
        config["brand"] = {r["Field Name"]: r["Value"] for r in brand_rows if r.get("Field Name")}
    except Exception as e:
        print(f"⚠️  Config_Brand load failed: {e}")

    print("🔌 Initializing Publisher Agent...")
    publisher = PublisherAgent(config=config)

    records = plan_sheet.get_all_records()
    to_publish = [
        (i + 2, row) for i, row in enumerate(records)
        if row.get("Status") in READY_STATUSES
        and row.get("Optimized_Draft")
        and not row.get("Published_Status")
    ]
    print(f"\n📝 Found {len(to_publish)} article(s) ready to publish.\n")

    published = 0
    for index, row in to_publish:
        title = row.get("Title", "")
        print(f"{'─'*60}\n🚀 [{published+1}/{len(to_publish)}] Publishing: {title}")

        optimizer_result = {
            "status":           "Ready for Approval",
            "html":             row.get("Optimized_Draft", ""),
            "schema":           row.get("Schema", ""),
            "meta_title":       row.get("MetaTitle", ""),
            "meta_description": row.get("MetaDescription", ""),
            "url_slug":         row.get("UrlSlug", ""),
            "word_count":       row.get("WordCount", 0),
            "warnings":         [],
        }

        try:
            result = publisher.publish_post(
                content_row=row,
                optimizer_result=optimizer_result,
                config=config,
            )
        except Exception as e:
            print(f"   ❌ Exception during publish: {e}")
            update_cells(plan_sheet, index, {"Reviewer_Notes": f"Publish error: {str(e)[:200]}"})
            continue

        pub_status = result.get("status", "")

        if pub_status in ("BLOCKED", "FAILED"):
            error_note = "; ".join(result.get("errors", []) + result.get("warnings", []))
            print(f"   ❌ FAILED: {error_note}")
            update_cells(plan_sheet, index, {"Reviewer_Notes": error_note})
            db.log_task("Publisher", title, pub_status, error_note)
            continue

        admin_url     = result.get("admin_url", "")
        is_hidden     = result.get("is_hidden", False)
        hidden_reason = result.get("hidden_reason", "")
        warning_note  = "; ".join(result.get("warnings", []))

        sheet_status    = "Live (Hidden)" if is_hidden else "Live"
        published_label = "Hidden - Needs Review" if is_hidden else "Live"

        update_cells(plan_sheet, index, {
            "Status":           sheet_status,
            "Published_Status": published_label,
            "AdminURL":         admin_url,
            "Reviewer_Notes":   warning_note or row.get("Reviewer_Notes", ""),
            "Visibility":       "hidden" if is_hidden else "public",
        })
        db.log_task("Publisher", title, "SUCCESS", f"{sheet_status} → {admin_url}")

        if chroma:
            try:
                chroma.save_post({
                    "title":          title,
                    "url_slug":       row.get("UrlSlug", ""),
                    "keyword":        row.get("Keyword", ""),
                    "section":        row.get("Section", ""),
                    "summary":        row.get("Summary", ""),
                    "published_date": row.get("ScheduledDate", ""),
                    "admin_url":      admin_url,
                })
                print("   🧠 Saved to ChromaDB.")
            except Exception as e:
                print(f"   ⚠️  ChromaDB save failed: {e}")

        send_telegram(title, admin_url, hidden=is_hidden, reason=hidden_reason)
        published += 1
        print(f"   ✅ {sheet_status}: {admin_url}")

    print(f"\n{'═'*60}")
    print(f"✅ Done. Published {published}/{len(to_publish)} articles.")


if __name__ == "__main__":
    main()
