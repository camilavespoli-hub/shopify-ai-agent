import os
import json
import gspread
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from database_manager import DatabaseManager

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "Glomend Blog AI — Dashboard",
    page_icon  = "🌿",
    layout     = "wide",
)

st.title("🌿 Glomend Blog AI — Pipeline Dashboard")
st.caption("Live view of content pipeline status, agent activity, and Shopify drafts.")

# ─────────────────────────────────────────────
# DATA LOADERS (cached to avoid re-fetching every second)
# ─────────────────────────────────────────────

@st.cache_resource(ttl=60)   # Reconnect at most once per minute
def get_sheet(sheet_name="Blog_agent_ai"):
    """Returns the Google Sheet object."""
    try:
        client = gspread.service_account(filename="service_account.json")
        return client.open(sheet_name)
    except Exception as e:
        st.error(f"❌ Google Sheets connection failed: {e}")
        return None


@st.cache_data(ttl=30)       # Refresh sheet data every 30 seconds
def load_content_plan(_sh):
    """Loads Content_Plan tab as a DataFrame."""
    if "RowID" in df_plan.columns:
        df_plan["RowID"] = df_plan["RowID"].replace("", None)
    try:
        ws      = _sh.worksheet("Content_Plan")
        records = ws.get_all_records()
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load Content_Plan: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_system_status(_sh):
    """Reads Config_System tab for System_Status value."""
    try:
        ws      = _sh.worksheet("Config_System")
        records = ws.get_all_records()
        for row in records:
            if row.get("Setting_Name") == "System_Status":
                return row.get("Setting_Value", "UNKNOWN")
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


@st.cache_data(ttl=60)
def load_products(_sh):
    """Loads Config_Products tab."""
    try:
        ws = _sh.worksheet("Config_Products")
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_cadence(_sh):
    """Loads Config_Cadence tab."""
    try:
        ws = _sh.worksheet("Config_Cadence")
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)      # DB refreshes every 10 seconds
def load_db_data():
    """Loads all data from SQLite via DatabaseManager."""
    db = DatabaseManager()
    return {
        "totals":    db.get_totals(),
        "by_agent":  db.get_agent_summary(),
        "breakdown": db.get_status_breakdown(),
        "logs":      db.get_recent_logs(limit=100),
        "runs":      db.get_recent_runs(limit=10),
    }


# ─────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────

sh        = get_sheet()
db_data   = load_db_data()
totals    = db_data["totals"]
df_plan   = load_content_plan(sh)   if sh else pd.DataFrame()
sys_status = load_system_status(sh) if sh else "UNKNOWN"
df_products = load_products(sh)     if sh else pd.DataFrame()
df_cadence  = load_cadence(sh)      if sh else pd.DataFrame()

# ─────────────────────────────────────────────
# AUTO-REFRESH TOGGLE
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Controls")
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
    if auto_refresh:
        st.info("Page will refresh automatically.")
        st.markdown(
            '<meta http-equiv="refresh" content="30">',
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader("📅 Publish Cadence")
    if not df_cadence.empty:
        active = df_cadence[df_cadence["Active"] == "Y"]
        if not active.empty:
            row = active.iloc[0]
            st.metric("Schedule",     row.get("PeriodType",   "—"))
            st.metric("Posts / Week", row.get("PostsPerWeek", "—"))
            st.metric("Publish Days", row.get("PublishDays",  "—"))
            st.metric("Time",         row.get("PublishTime",  "—"))
    else:
        st.caption("Config_Cadence not loaded.")

    st.divider()
    st.subheader("💊 Products")
    if not df_products.empty:
        for _, p in df_products.iterrows():
            st.markdown(f"**{p.get('Name', '—')}**")
            st.caption(p.get("ShortDescription", ""))
    else:
        st.caption("Config_Products not loaded.")

# ─────────────────────────────────────────────
# ROW 1: SYSTEM STATUS + KPI CARDS
# ─────────────────────────────────────────────

status_color = {
    "ACTIVE":  "🟢",
    "PAUSED":  "🟡",
    "STOPPED": "🔴",
}.get(sys_status, "⚪")

st.subheader(f"System Status: {status_color} {sys_status}")
st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Tasks Logged", totals.get("total",       0))
k2.metric("✅ Successes",        totals.get("successes",   0))
k3.metric("❌ Failures",         totals.get("failures",    0))
k4.metric("⚠️ Needs Review",     totals.get("needs_review",0))

st.divider()

# ─────────────────────────────────────────────
# ROW 2: CONTENT PLAN STATUS + AGENT BAR CHART
# ─────────────────────────────────────────────

col_plan, col_bar = st.columns([1.6, 1.4])

with col_plan:
    st.subheader("📋 Content Plan — Status Overview")
    if not df_plan.empty and "Status" in df_plan.columns:
        status_counts = (
            df_plan["Status"]
            .replace("", "No Status")
            .value_counts()
            .reset_index()
        )
        status_counts.columns = ["Status", "Count"]

        status_color_map = {
            "Pending Approval": "#f59e0b",
            "Content Approved": "#3b82f6",
            "Ready to Publish": "#8b5cf6",
            "Live":             "#10b981",
            "Needs Review":     "#ef4444",
            "No Status":        "#9ca3af",
        }

        fig_pie = px.pie(
            status_counts,
            values     = "Count",
            names      = "Status",
            color      = "Status",
            color_discrete_map = status_color_map,
            hole       = 0.45,
        )
        fig_pie.update_layout(
            margin    = dict(t=10, b=10, l=10, r=10),
            height    = 300,
            showlegend= True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # KPIs below pie
        p1, p2, p3 = st.columns(3)
        p1.metric("Total Articles",   len(df_plan))
        p2.metric("Live",
                  len(df_plan[df_plan["Status"] == "Live"])
                  if "Status" in df_plan.columns else 0)
        p3.metric("Pending Approval",
                  len(df_plan[df_plan["Status"] == "Pending Approval"])
                  if "Status" in df_plan.columns else 0)
    else:
        st.info("No Content_Plan data available.")

with col_bar:
    st.subheader("🤖 Agent Performance")
    agent_data = db_data["by_agent"]
    if agent_data:
        df_agents = pd.DataFrame(agent_data)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Successes",
            x=df_agents["agent"],
            y=df_agents["successes"],
            marker_color="#10b981"
        ))
        fig_bar.add_trace(go.Bar(
            name="Needs Review",
            x=df_agents["agent"],
            y=df_agents["needs_review"],
            marker_color="#f59e0b"
        ))
        fig_bar.add_trace(go.Bar(
            name="Failures",
            x=df_agents["agent"],
            y=df_agents["failures"],
            marker_color="#ef4444"
        ))
        fig_bar.update_layout(
            barmode       = "stack",
            height        = 300,
            margin        = dict(t=10, b=10, l=10, r=10),
            legend        = dict(orientation="h", y=-0.25),
            xaxis_title   = "",
            yaxis_title   = "Tasks",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No agent activity logged yet.")

st.divider()

# ─────────────────────────────────────────────
# ROW 3: CONTENT PLAN TABLE
# ─────────────────────────────────────────────

st.subheader("📄 Content Plan — Full Table")

if not df_plan.empty:
    # Filter controls
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        status_filter = st.multiselect(
            "Filter by Status",
            options = sorted(df_plan["Status"].unique().tolist())
                      if "Status" in df_plan.columns else [],
            default = []
        )
    with f2:
        section_filter = st.multiselect(
            "Filter by Section",
            options = sorted(df_plan["Section"].unique().tolist())
                      if "Section" in df_plan.columns else [],
            default = []
        )
    with f3:
        search_term = st.text_input("🔍 Search Title", "")

    # Apply filters
    filtered = df_plan.copy()
    if status_filter:
        filtered = filtered[filtered["Status"].isin(status_filter)]
    if section_filter:
        filtered = filtered[filtered["Section"].isin(section_filter)]
    if search_term:
        filtered = filtered[
            filtered["Title"].str.contains(search_term, case=False, na=False)
        ]

    # Show only key columns — skip large content blobs
    display_cols = [
        c for c in [
            "RowID", "Status", "Title", "Section", "Keyword",
            "ScheduledDate", "TrendScore", "WordCount",
            "MetaTitle", "AdminURL", "Published_Status"
        ] if c in filtered.columns
    ]
    st.dataframe(
        filtered[display_cols],
        use_container_width = True,
        height              = 320,
    )
    st.caption(f"Showing {len(filtered)} of {len(df_plan)} rows")
else:
    st.info("Content_Plan is empty or could not be loaded.")

st.divider()

# ─────────────────────────────────────────────
# ROW 4: PIPELINE RUN HISTORY + RECENT LOGS
# ─────────────────────────────────────────────

col_runs, col_logs = st.columns([1, 2])

with col_runs:
    st.subheader("🕐 Recent Pipeline Runs")
    runs = db_data["runs"]
    if runs:
        df_runs = pd.DataFrame(runs)
        df_runs["duration"] = df_runs.apply(
            lambda r: (
                pd.to_datetime(r["finished_at"]) -
                pd.to_datetime(r["started_at"])
            ).seconds // 60
            if r.get("finished_at") else "Running",
            axis=1
        )
        st.dataframe(
            df_runs[["id", "started_at", "success_count",
                      "fail_count", "duration", "summary_note"]],
            use_container_width = True,
            height              = 280,
        )
    else:
        st.info("No pipeline runs recorded yet.")

with col_logs:
    st.subheader("📜 Recent Task Log")
    logs = db_data["logs"]
    if logs:
        df_logs  = pd.DataFrame(logs)
        agent_opt = ["All"] + sorted(df_logs["agent"].unique().tolist())
        log_agent = st.selectbox("Filter by Agent", agent_opt, key="log_agent")
        if log_agent != "All":
            df_logs = df_logs[df_logs["agent"] == log_agent]

        # Colour-code status
        def colour_status(val):
            colours = {
                "SUCCESS":      "background-color: #d1fae5",
                "BLOCKED":      "background-color: #fee2e2",
                "FAILED":       "background-color: #fee2e2",
                "HARD_FAIL":    "background-color: #fee2e2",
                "NEEDS REVIEW": "background-color: #fef3c7",
            }
            return colours.get(val, "")

        styled = (
            df_logs[["timestamp", "agent", "title", "status", "note"]]
            .style
            .applymap(colour_status, subset=["status"])
        )
        st.dataframe(styled, use_container_width=True, height=280)
    else:
        st.info("No task logs yet.")

st.divider()

# ─────────────────────────────────────────────
# ROW 5: PUBLISHED ARTICLES
# ─────────────────────────────────────────────

st.subheader("🚀 Published Drafts")

if not df_plan.empty and "Published_Status" in df_plan.columns:
    published = df_plan[df_plan["Published_Status"].astype(str).str.len() > 0]
    if not published.empty:
        pub_cols = [
            c for c in ["Title", "Section", "Keyword",
                         "ScheduledDate", "AdminURL", "Published_Status"]
            if c in published.columns
        ]
        # Make AdminURL clickable
        if "AdminURL" in published.columns:
            published = published.copy()
            published["AdminURL"] = published["AdminURL"].apply(
                lambda u: f'<a href="{u}" target="_blank">Review in Shopify</a>'
                          if u else ""
            )
            st.markdown(
                published[pub_cols].to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
        else:
            st.dataframe(published[pub_cols], use_container_width=True)
    else:
        st.info("No published articles yet.")
else:
    st.info("Published_Status column not found — add it to Content_Plan.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.caption("Glomend Blog AI Pipeline · Dashboard · Data refreshes every 30s")
