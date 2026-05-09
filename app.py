import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import uuid
import sys
import os
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

sys.path.append(os.path.dirname(__file__))

from ai_agent import run_query, llm
from analytics import (
    get_total_revenue,
    get_total_orders,
    get_total_customers,
    get_total_products,
    get_monthly_revenue,
    get_weekly_revenue,
    get_city_sales,
    get_category_sales,
    get_top_customers,
    get_order_status,
    get_avg_order_value,
    get_new_customers_this_month,
    get_repeat_customer_rate,
    get_top_products,
    get_customer_segments,
    get_top_customers_enhanced,
    get_fulfillment_rate,
)
from chat_db import (
    init_chat_table,
    create_session,
    rename_session,
    delete_session,
    load_all_sessions,
    save_chat,
    load_chat,
    load_conversation_context,
    clear_chat
)

# ════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ════════════════════════════════════════════════════════════════

def similarity(a, b):
    emb1 = model.encode(str(a), convert_to_tensor=True)
    emb2 = model.encode(str(b), convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def keyword_score(answer, keywords):
    answer = str(answer).lower()
    matched = sum(1 for k in keywords if k in answer)
    return matched / len(keywords) if keywords else 1.0

def final_score_calc(semantic, keyword):
    return 0.7 * semantic + 0.3 * keyword

def accuracy_percent(expected, predicted):
    try:
        error_pct = abs(expected - predicted) / expected * 100
        return round(max(0, 100 - error_pct), 2)
    except:
        return 0

# ─── AGENT TEST SUITE ────────────────────────────────────────────
# These are the ground-truth test cases for Agent Task Success Rate
# Each question has:
#   - expected_keywords: words the answer MUST contain to be correct
#   - expected_data_check: a lambda to verify the returned data is correct
AGENT_TEST_SUITE = [
    {
        "question": "What is the total revenue?",
        "expected_answer": "The total revenue of the e-commerce store is shown with exact amount.",
        "keywords": ["revenue", "total"],
        "data_check": lambda d: isinstance(d, (int, float, list)) and d is not None,
        "category": "Aggregation"
    },
    {
        "question": "How many total orders are there?",
        "expected_answer": "The total number of orders placed in the system.",
        "keywords": ["orders", "total"],
        "data_check": lambda d: d is not None,
        "category": "Aggregation"
    },
    {
        "question": "Who are the top 5 customers?",
        "expected_answer": "List of top customers ranked by order value or frequency.",
        "keywords": ["customer", "top"],
        "data_check": lambda d: isinstance(d, list) and len(d) > 0,
        "category": "Ranking"
    },
    {
        "question": "Show me revenue by category",
        "expected_answer": "Revenue breakdown across different product categories.",
        "keywords": ["category", "revenue"],
        "data_check": lambda d: isinstance(d, list) and len(d) > 0,
        "category": "Grouping"
    },
    {
        "question": "What is the order status breakdown?",
        "expected_answer": "Summary of orders by status such as delivered, pending, cancelled.",
        "keywords": ["status", "order"],
        "data_check": lambda d: d is not None,
        "category": "Status"
    },
    {
        "question": "Show monthly revenue trend",
        "expected_answer": "Monthly revenue figures showing trend over time.",
        "keywords": ["monthly", "revenue", "trend"],
        "data_check": lambda d: isinstance(d, list) and len(d) > 0,
        "category": "Trend"
    },
    {
        "question": "How many customers do we have?",
        "expected_answer": "Total number of unique customers in the system.",
        "keywords": ["customers"],
        "data_check": lambda d: d is not None,
        "category": "Aggregation"
    },
    {
        "question": "What are the sales by city?",
        "expected_answer": "Sales or revenue data segmented by city.",
        "keywords": ["city", "sales"],
        "data_check": lambda d: isinstance(d, list) and len(d) > 0,
        "category": "Grouping"
    },
]

def run_agent_evaluation():
    """Run all test cases through the agent and score them."""
    results = []
    for tc in AGENT_TEST_SUITE:
        try:
            result = run_query(tc["question"], conversation_context=[])
            if "error" in result:
                agent_answer = ""
                raw_data = None
                error = result["error"]
            else:
                agent_answer = result.get("insight", "")
                raw_data = result.get("data", None)
                error = None

            # 1. Semantic similarity score (0-1)
            sem = similarity(tc["expected_answer"], agent_answer) if agent_answer else 0.0

            # 2. Keyword match score (0-1)
            kw = keyword_score(agent_answer, tc["keywords"])

            # 3. Task success — did agent return valid data?
            try:
                task_success = 1 if tc["data_check"](raw_data) else 0
            except:
                task_success = 0

            # 4. Hallucination flag — answer references data NOT from DB
            hallucination = 0
            if agent_answer and raw_data is None and "error" not in str(agent_answer).lower():
                hallucination = 1

            # 5. Final composite score
            final = round(0.4 * sem + 0.3 * kw + 0.3 * task_success, 3)

            # 6. Pass/Fail (threshold 0.55)
            passed = "✅ Pass" if final >= 0.55 else "❌ Fail"

            results.append({
                "Question": tc["question"],
                "Category": tc["category"],
                "Semantic Score": round(sem, 3),
                "Keyword Score": round(kw, 3),
                "Task Success": task_success,
                "Hallucination": hallucination,
                "Final Score": final,
                "Result": passed,
                "Agent Answer": agent_answer[:120] + "..." if len(str(agent_answer)) > 120 else agent_answer,
                "Error": error or ""
            })
        except Exception as e:
            results.append({
                "Question": tc["question"],
                "Category": tc["category"],
                "Semantic Score": 0,
                "Keyword Score": 0,
                "Task Success": 0,
                "Hallucination": 0,
                "Final Score": 0,
                "Result": "❌ Fail",
                "Agent Answer": "",
                "Error": str(e)
            })
    return results

def compute_precision_recall_f1(results_df):
    """
    Treat Pass=1, Fail=0 as binary classification.
    Ground truth = all should pass (1).
    Predicted = agent's pass/fail.
    """
    y_true = [1] * len(results_df)
    y_pred = [1 if "Pass" in r else 0 for r in results_df["Result"]]

    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)

# ════════════════════════════════════════════════════════════════
# INIT
# ════════════════════════════════════════════════════════════════
init_chat_table()

st.set_page_config(
    page_title="E-Commerce AI Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #111318 !important;
            border-right: 1px solid #1e2130;
        }
        [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        .new-chat-btn button {
            background: linear-gradient(135deg, #2e86de, #1a6fc4) !important;
            color: white !important; border: none !important;
            border-radius: 10px !important; font-weight: 700 !important;
            width: 100% !important; padding: 12px !important;
            font-size: 15px !important; letter-spacing: 0.3px !important;
            margin-bottom: 8px !important;
        }
        .session-item button {
            background-color: transparent !important; color: #cccccc !important;
            border: 1px solid transparent !important; border-radius: 8px !important;
            text-align: left !important; width: 100% !important;
            padding: 9px 12px !important; font-size: 13.5px !important;
            transition: all 0.15s !important;
        }
        .session-item button:hover {
            background-color: #1e2130 !important;
            border-color: #2a2d3e !important; color: white !important;
        }
        .session-item-active button {
            background-color: #1e2d45 !important; border: 1px solid #2e86de !important;
            color: white !important; border-radius: 8px !important;
            text-align: left !important; width: 100% !important;
            padding: 9px 12px !important; font-size: 13.5px !important;
        }
        .delete-btn button {
            background-color: transparent !important; color: #666 !important;
            border: none !important; font-size: 12px !important; padding: 4px 8px !important;
        }
        .delete-btn button:hover { color: #e74c3c !important; background-color: transparent !important; }
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background-color: #2e86de !important; color: white !important;
            border-radius: 10px !important; font-size: 15px !important; font-weight: 700 !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background-color: #1a1d27 !important; color: #aaaaaa !important;
            border: 1px solid #2a2d3e !important; border-radius: 10px !important;
            font-size: 15px !important; font-weight: 600 !important;
        }
        .metric-card {
            background-color: #1a1d27; border-radius: 12px; padding: 20px;
            border: 1px solid #2a2d3e; text-align: center;
        }
        .metric-value { font-size: 28px; font-weight: 700; color: #2e86de; }
        .metric-label { font-size: 13px; color: #aaaaaa; margin-top: 4px; }
        .eval-metric-card {
            background-color: #1a1d27; border-radius: 12px; padding: 18px;
            border: 1px solid #2a2d3e; text-align: center; margin: 4px 0;
        }
        .eval-metric-value { font-size: 32px; font-weight: 800; color: #7ed321; }
        .eval-metric-label { font-size: 13px; color: #aaaaaa; margin-top: 4px; }
        .user-message {
            background-color: #2e86de; color: white; padding: 12px 16px;
            border-radius: 18px 18px 4px 18px; margin: 8px 0;
            max-width: 75%; margin-left: auto; text-align: right;
        }
        .ai-message {
            background-color: #1a1d27; color: #e0e0e0; padding: 12px 16px;
            border-radius: 18px 18px 18px 4px; margin: 8px 0;
            max-width: 85%; border: 1px solid #2a2d3e;
        }
        .ai-data {
            background-color: #0f1117; color: #7ed321; padding: 10px 14px;
            border-radius: 8px; font-family: 'Courier New', monospace;
            font-size: 14px; margin-top: 8px; border: 1px solid #2a2d3e; white-space: pre-wrap;
        }
        .insight-box {
            margin-top: 12px; padding: 12px;
            background-color: rgba(46, 134, 222, 0.1);
            border-left: 4px solid #2e86de; border-radius: 4px; color: #ffffff;
        }
        .stButton button {
            background-color: #2e86de; color: white; border: none;
            border-radius: 10px; padding: 10px 24px; font-weight: 600;
        }
        .stButton button:hover { background-color: #1a6fc4; }
        .sidebar-section-label {
            font-size: 11px; color: #555577; text-transform: uppercase;
            letter-spacing: 1px; margin: 12px 0 6px 4px;
        }
    </style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
def switch_session(sid):
    st.session_state.session_id = sid
    st.session_state.chat_history = load_chat(sid)
    st.session_state.conversation_context = load_conversation_context(sid, last_n=6)

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "dashboard"

if st.session_state.active_tab == "chat":
    st.markdown("""<style>
        [data-testid="stSidebar"] { display: block !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💬 Chat History")
    st.markdown("---")
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✏️  New Chat", key="new_chat_btn", use_container_width=True):
        new_sid = str(uuid.uuid4())
        create_session(new_sid, title="New Chat")
        switch_session(new_sid)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Recent Chats</div>', unsafe_allow_html=True)
    all_sessions = load_all_sessions()

    for s in all_sessions:
        sid       = s["session_id"]
        title     = s["title"] or "New Chat"
        is_active = sid == st.session_state.get("session_id", "")
        col_title, col_del = st.columns([5, 1])
        with col_title:
            css_class = "session-item-active" if is_active else "session-item"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            label = f"{'▶ ' if is_active else ''}{title[:38]}{'…' if len(title) > 38 else ''}"
            if st.button(label, key=f"sess_{sid}"):
                switch_session(sid)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col_del:
            st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
            if st.button("🗑", key=f"del_{sid}", help="Delete this chat"):
                delete_session(sid)
                if sid == st.session_state.get("session_id"):
                    remaining = [x for x in all_sessions if x["session_id"] != sid]
                    if remaining:
                        switch_session(remaining[0]["session_id"])
                    else:
                        new_sid = str(uuid.uuid4())
                        create_session(new_sid)
                        switch_session(new_sid)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if not all_sessions:
        st.markdown('<div style="color:#444;font-size:13px;padding:8px 4px;">No conversations yet.</div>', unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────
if "session_id" not in st.session_state:
    sessions = load_all_sessions()
    if sessions:
        st.session_state.session_id = sessions[0]["session_id"]
    else:
        new_sid = str(uuid.uuid4())
        create_session(new_sid)
        st.session_state.session_id = new_sid

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat(st.session_state.session_id)

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = load_conversation_context(
        st.session_state.session_id, last_n=6
    )

# ─── HEADER ──────────────────────────────────────────────────────
st.markdown("## 🛒 E-Commerce AI Intelligence System")
st.markdown("---")

# ─── TAB SWITCHER ────────────────────────────────────────────────
col_t1, col_t2, col_t3, col_spacer = st.columns([1.2, 1.2, 1.5, 7])
with col_t1:
    if st.button("📊 Dashboard", key="tab_dash",
                 type="primary" if st.session_state.active_tab == "dashboard" else "secondary",
                 use_container_width=True):
        st.session_state.active_tab = "dashboard"
        st.rerun()
with col_t2:
    if st.button("🤖 Ask AI", key="tab_chat",
                 type="primary" if st.session_state.active_tab == "chat" else "secondary",
                 use_container_width=True):
        st.session_state.active_tab = "chat"
        st.rerun()
with col_t3:
    if st.button("📊 Model Eval", key="tab_eval",
                 type="primary" if st.session_state.active_tab == "evaluation" else "secondary",
                 use_container_width=True):
        st.session_state.active_tab = "evaluation"
        st.rerun()
st.markdown("---")


# ════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD (Enhanced)
# ════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "dashboard":

    st.markdown("### 📈 Business Overview")

    # ── Row 1: Core KPIs ────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total_revenue   = get_total_revenue()
    total_orders    = get_total_orders()
    total_customers = get_total_customers()
    total_products  = get_total_products()

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">₹ {total_revenue:,.2f}</div>
            <div class="metric-label">Total Revenue</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total_orders:,}</div>
            <div class="metric-label">Total Orders</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">Total Customers</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total_products:,}</div>
            <div class="metric-label">Total Products</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: New KPIs ─────────────────────────────────────────
    avg_order_val      = get_avg_order_value()
    new_customers      = get_new_customers_this_month()
    repeat_rate        = get_repeat_customer_rate()
    fulfillment_rate   = get_fulfillment_rate()

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">₹ {avg_order_val:,.2f}</div>
            <div class="metric-label">Avg. Order Value</div></div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{new_customers:,}</div>
            <div class="metric-label">New Customers (This Month)</div></div>""", unsafe_allow_html=True)
    with col7:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{repeat_rate:.1f}%</div>
            <div class="metric-label">Repeat Customer Rate</div></div>""", unsafe_allow_html=True)
    with col8:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{fulfillment_rate:.1f}%</div>
            <div class="metric-label">Fulfillment Rate</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Monthly Revenue Trend + Weekly Revenue ───────────────────
    col_trend, col_weekly = st.columns([2, 1])

    with col_trend:
        st.markdown("### 📅 Monthly Revenue Trend")
        revenue_df = get_monthly_revenue()
        fig_line = px.line(revenue_df, x="Month", y="Revenue", markers=True,
                           color_discrete_sequence=["#2e86de"])
        fig_line.update_layout(
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#ffffff",
            xaxis=dict(gridcolor="#2a2d3e"), yaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_weekly:
        st.markdown("### 📊 Weekly Revenue")
        weekly_df = get_weekly_revenue()
        fig_weekly = px.bar(weekly_df, x="Week_Start", y="Revenue",
                            color="Revenue",
                            color_continuous_scale=["#1a6fc4", "#2e86de", "#7ed321"])
        fig_weekly.update_layout(
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#ffffff",
            xaxis=dict(gridcolor="#2a2d3e", tickangle=-45),
            yaxis=dict(gridcolor="#2a2d3e"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=40), height=280
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

    st.markdown("---")

    # ── Region, Category, Order Status ──────────────────────────
    col_reg, col_cat, col_status = st.columns(3)

    with col_reg:
        st.markdown("### 🏙️ City-wise Sales")
        city_df = get_city_sales()
        fig_pie = px.bar(city_df, x="Revenue", y="City", orientation="h",
                         color="Revenue", color_continuous_scale=["#1a6fc4","#2e86de","#7ed321"],
                         labels={"Revenue": "Revenue (₹)", "City": "City"})
        fig_pie.update_layout(plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
                              font_color="#ffffff", xaxis=dict(gridcolor="#2a2d3e"),
                              yaxis=dict(gridcolor="#2a2d3e"), coloraxis_showscale=False,
                              margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_cat:
        st.markdown("### 📦 Category Revenue")
        category_df = get_category_sales()
        fig_bar = px.bar(category_df, x="Revenue", y="Category", orientation="h",
                         color_discrete_sequence=["#2e86de"])
        fig_bar.update_layout(plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
                              font_color="#ffffff", xaxis=dict(gridcolor="#2a2d3e"),
                              yaxis=dict(gridcolor="#2a2d3e"),
                              margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_status:
        st.markdown("### 📋 Order Status")
        status = get_order_status()
        fulfil = get_fulfillment_rate()
        fig_donut = go.Figure(data=[go.Pie(
            labels=status["Status"], values=status["Count"],
            hole=0.55, marker_colors=["#e74c3c","#7ed321","#f39c12",]
        )])
        fig_donut.update_layout(
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
            font_color="#ffffff", margin=dict(l=10,r=10,t=10,b=10), height=240
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown(f"""<div style="text-align:center; margin-top:-10px;">
            <span style="font-size:13px; color:#aaaaaa;">Fulfillment Rate: </span>
            <span style="font-size:15px; font-weight:700; color:#7ed321;">{fulfil:.1f}%</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Top Products + Customer Segments ────────────────────────
    col_prod, col_seg = st.columns(2)

    with col_prod:
        st.markdown("### 🏆 Top Products by Revenue")
        top_products_df = get_top_products(limit=5)
        top_products_df["Revenue"] = top_products_df["Revenue"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(top_products_df, use_container_width=True, hide_index=True)

    with col_seg:
        st.markdown("### 👥 Customer Segments")
        seg_df = get_customer_segments()
        seg_colors = {
            "Champions": "#2e86de",
            "Loyal":     "#7ed321",
            "Potential": "#f39c12",
            "At-risk":   "#888780",
            "Lost":      "#e74c3c",
        }
        colors = [seg_colors.get(s, "#2e86de") for s in seg_df["Segment"]]
        fig_seg = go.Figure(data=[go.Pie(
            labels=seg_df["Segment"], values=seg_df["Count"],
            hole=0.5, marker_colors=colors, textinfo="label+percent"
        )])
        fig_seg.update_layout(
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
            font_color="#ffffff", margin=dict(l=10,r=10,t=10,b=10), height=270,
            showlegend=False
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("---")

    # ── Enhanced Top Customers ───────────────────────────────────
    st.markdown("### 👥 Top 10 Customers")
    top_customers_df = get_top_customers_enhanced(limit=10)
    st.dataframe(top_customers_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# AUTO CHART HELPERS
# ════════════════════════════════════════════════════════════════
def detect_chart_type(user_query: str, data: list) -> str:
    if not isinstance(data, list) or len(data) < 2:
        return "none"
    keys = list(data[0].keys()) if data else []
    if len(keys) < 2:
        return "none"
    prompt = f"""
You are a data visualization expert. Choose the BEST chart type. Reply with ONLY one word:
- "bar"  → comparing categories
- "line" → trends over time
- "pie"  → proportions/percentage
- "none" → not chartable

Question: "{user_query}"
Data: {json.dumps(data[:2], default=str)}
"""
    result = llm.invoke(prompt).content.strip().lower()
    return result if result in ("bar", "line", "pie") else "none"


def render_chart(data: list, chart_type: str, title: str):
    df = pd.DataFrame(data)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    cols = df.columns.tolist()
    label_col, value_col = cols[0], cols[1]
    BASE = dict(
        plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
        font_color="#ffffff", margin=dict(l=20, r=20, t=45, b=20),
        height=370,
        title=dict(text=f"📊 {title}", font=dict(size=15, color="#ffffff")),
        xaxis=dict(gridcolor="#2a2d3e"),
        yaxis=dict(gridcolor="#2a2d3e"),
    )
    if chart_type == "bar":
        fig = px.bar(df, x=label_col, y=value_col, color=value_col,
                     color_continuous_scale="Blues", template="plotly_dark")
        fig.update_layout(**BASE, coloraxis_showscale=False)
    elif chart_type == "line":
        fig = px.line(df, x=label_col, y=value_col, markers=True, template="plotly_dark")
        fig.update_traces(line_color="#2e86de", marker_color="#64B5F6")
        fig.update_layout(**BASE)
    elif chart_type == "pie":
        fig = px.pie(df, names=label_col, values=value_col, template="plotly_dark")
        fig.update_layout(**BASE)
    else:
        return
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — ASK AI
# ════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "chat":
    all_sessions = load_all_sessions()
    active = next((s for s in all_sessions if s["session_id"] == st.session_state.session_id), None)
    active_title = active["title"] if active else "New Chat"

    st.markdown(f"### 🤖 {active_title}")

    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="user-message">💬 {chat["query"]}</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="ai-message">
                🤖 <b>Data Result:</b>
                <div class="ai-data">{chat["data"]}</div>
                <div class="insight-box">
                    <b style="color: #2e86de;">💡 AI Insight:</b><br>{chat["insight"]}
                </div>
            </div>""", unsafe_allow_html=True)
        if chat.get("chart_type", "none") != "none" and chat.get("chart_data"):
            render_chart(chat["chart_data"], chat["chart_type"], chat["query"].capitalize())

    # ─── DOWNLOAD REPORT ──────────────────────────────────────
    if "last_result" in st.session_state and isinstance(st.session_state["last_result"], list):
        st.success("✅ Report ready for download")
        st.markdown("### 📥 Download Report")
        df = pd.DataFrame(st.session_state["last_result"])
        col1, col2 = st.columns(2)
        with col1:
            excel_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            df.to_excel(excel_file.name, index=False)
            with open(excel_file.name, "rb") as f:
                st.download_button(label="📊 Download Excel", data=f,
                    file_name="AI_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col2:
            def generate_pdf():
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc = SimpleDocTemplate(temp_pdf.name)
                styles = getSampleStyleSheet()
                elements = []
                elements.append(Paragraph("AI E-Commerce Intelligence Report", styles['Title']))
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(f"<b>Query:</b> {st.session_state.get('last_query','')}", styles['Normal']))
                elements.append(Spacer(1, 10))
                data = st.session_state["last_result"]
                table_data = [list(data[0].keys())] + [list(d.values()) for d in data]
                elements.append(Table(table_data))
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(f"<b>AI Insight:</b> {st.session_state.get('last_insight','')}", styles['Normal']))
                doc.build(elements)
                return temp_pdf.name
            pdf_path = generate_pdf()
            with open(pdf_path, "rb") as f:
                st.download_button(label="📄 Download PDF", data=f,
                    file_name="AI_Report.pdf", mime="application/pdf")

    # ─── INPUT ────────────────────────────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input("", placeholder="Ask a business question...", label_visibility="collapsed")
        with col_btn:
            submit = st.form_submit_button("Ask 🚀")

    if submit and user_input:
        create_session(st.session_state.session_id)
        with st.spinner("🔍 Analysing..."):
            result = run_query(user_input, conversation_context=st.session_state.conversation_context)

        if "error" in result:
            st.error(result["error"])
        else:
            raw_data = result.get("data", "No data")
            insight  = result.get("insight", "No insight")
            st.session_state["last_result"] = raw_data
            st.session_state["last_query"]  = user_input
            st.session_state["last_insight"] = insight

            # Live evaluation tracking (for the eval tab's live section)
            if "live_eval_data" not in st.session_state:
                st.session_state["live_eval_data"] = []

            gt_map = {
                "total revenue":    {"answer": "The total revenue is the sum of all order amounts.", "keywords": ["revenue", "total"]},
                "total orders":     {"answer": "The total number of orders placed.", "keywords": ["orders", "total"]},
                "top customers":    {"answer": "Top customers ranked by spending.", "keywords": ["customers", "top"]},
                "revenue by category": {"answer": "Revenue split across categories.", "keywords": ["category", "revenue"]},
                "order status":     {"answer": "Orders grouped by their status.", "keywords": ["status", "order"]},
                "monthly revenue":  {"answer": "Revenue trend over months.", "keywords": ["monthly", "revenue"]},
                "sales by city":    {"answer": "Sales data by city.", "keywords": ["city", "sales"]},
            }
            gt = next((v for k, v in gt_map.items() if k in user_input.lower()), None)
            expected_answer = gt["answer"] if gt else insight
            keywords        = gt["keywords"] if gt else []

            sem   = similarity(expected_answer, insight)
            kw    = keyword_score(insight, keywords)
            final = final_score_calc(sem, kw)

            task_success = 1 if raw_data and raw_data != "No data" else 0

            st.session_state["live_eval_data"].append({
                "Question":       user_input,
                "Semantic Score": round(sem, 3),
                "Keyword Score":  round(kw, 3),
                "Task Success":   task_success,
                "Final Score":    round(final, 3),
                "Result":         "✅ Pass" if final >= 0.55 else "❌ Fail"
            })

            chart_type = detect_chart_type(user_input, raw_data) if isinstance(raw_data, list) and len(raw_data) >= 2 else "none"
            save_chat(st.session_state.session_id, user_input, raw_data, insight, chart_type,
                      raw_data if chart_type != "none" else [])
            st.session_state.chat_history.append({
                "query": user_input, "data": raw_data,
                "insight": insight, "chart_type": chart_type, "chart_data": raw_data
            })
            st.session_state.conversation_context.extend([
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": insight}
            ])
            st.rerun()


# ════════════════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION 
# ════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "evaluation":

    st.markdown("## 📊 Model Evaluation")
    st.markdown("---")

    # ════════════════════════════════════
    # SECTION 5.1 + 5.2 + 5.3 — RUN TEST SUITE
    # ════════════════════════════════════
    st.markdown("### 🧪 Agent Test Suite Evaluation")
    st.markdown("Run **8 predefined test questions** through your AI agent and measure performance using standard ML metrics.")

    col_run, col_clear = st.columns([2, 8])
    with col_run:
        run_eval = st.button("▶️ Run Full Evaluation", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑 Clear Results", use_container_width=False):
            if "eval_results" in st.session_state:
                del st.session_state["eval_results"]
            st.rerun()

    if run_eval:
        with st.spinner("🔍 Running all test cases through the AI agent... This may take a minute."):
            results = run_agent_evaluation()
            st.session_state["eval_results"] = results
        st.success("✅ Evaluation complete!")
        st.rerun()

    if "eval_results" in st.session_state and st.session_state["eval_results"]:
        results = st.session_state["eval_results"]
        df_eval = pd.DataFrame(results)

        # ── 5.1 Model Performance Metrics ─────────────────────
        st.markdown("---")
        st.markdown("#### 📐 5.1 — Model Performance (Accuracy, Precision, Recall, F1)")

        accuracy, precision, recall, f1 = compute_precision_recall_f1(df_eval)
        pass_count  = sum(1 for r in results if "Pass" in r["Result"])
        fail_count  = len(results) - pass_count
        task_success_rate = round(sum(r["Task Success"] for r in results) / len(results) * 100, 1)
        hallucination_rate = round(sum(r["Hallucination"] for r in results) / len(results) * 100, 1)
        avg_semantic = round(df_eval["Semantic Score"].mean(), 3)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value">{accuracy*100:.1f}%</div>
                <div class="eval-metric-label">🎯 Accuracy</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value">{precision*100:.1f}%</div>
                <div class="eval-metric-label">🔍 Precision</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value">{recall*100:.1f}%</div>
                <div class="eval-metric-label">📡 Recall</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value">{f1*100:.1f}%</div>
                <div class="eval-metric-label">⚖️ F1 Score</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 5.2 Agent Task Success Rate ────────────────────────
        st.markdown("#### 🤖 5.2 — Agent Task Success Rate")
        m5, m6, m7, m8 = st.columns(4)
        with m5:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value">{task_success_rate}%</div>
                <div class="eval-metric-label">✅ Task Success Rate</div></div>""", unsafe_allow_html=True)
        with m6:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value">{pass_count}/{len(results)}</div>
                <div class="eval-metric-label">🟢 Tests Passed</div></div>""", unsafe_allow_html=True)
        with m7:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value" style="color:#e74c3c">{hallucination_rate}%</div>
                <div class="eval-metric-label">🚨 Hallucination Rate</div></div>""", unsafe_allow_html=True)
        with m8:
            st.markdown(f"""<div class="eval-metric-card">
                <div class="eval-metric-value">{avg_semantic}</div>
                <div class="eval-metric-label">🧠 Avg Semantic Score</div></div>""", unsafe_allow_html=True)

        # Overall status banner
        st.markdown("<br>", unsafe_allow_html=True)
        if f1 >= 0.75:
            st.success(f"✅ Agent is performing well — F1 Score: {f1*100:.1f}%")
        elif f1 >= 0.5:
            st.warning(f"⚠️ Agent is performing moderately — F1 Score: {f1*100:.1f}%. Consider improving prompts or adding more data.")
        else:
            st.error(f"❌ Agent needs improvement — F1 Score: {f1*100:.1f}%")

        # ── 5.3 Response Quality Analysis ─────────────────────
        st.markdown("---")
        st.markdown("#### 🔬 5.3 — Response Quality Analysis")
        st.markdown("Detailed breakdown of each test case — semantic relevance, keyword match, task completion, and hallucination detection.")

        display_cols = ["Question", "Category", "Semantic Score", "Keyword Score",
                        "Task Success", "Hallucination", "Final Score", "Result"]
        st.dataframe(df_eval[display_cols], use_container_width=True, hide_index=True)

        with st.expander("📄 View Full Agent Answers"):
            for i, row in df_eval.iterrows():
                st.markdown(f"**Q{i+1}: {row['Question']}**")
                st.markdown(f"🤖 Agent: {row['Agent Answer']}")
                if row['Error']:
                    st.error(f"Error: {row['Error']}")
                st.markdown("---")

        # ── 5.4 Visualizations ────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 5.4 — Visualizations")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Bar chart: Final Score per question
            df_chart = df_eval.copy()
            df_chart["Short Q"] = df_chart["Question"].str[:25] + "..."
            fig_scores = px.bar(
                df_chart, x="Short Q", y="Final Score",
                color="Final Score", color_continuous_scale=["#e74c3c", "#f39c12", "#7ed321"],
                title="Agent Final Score per Test Question",
                range_y=[0, 1]
            )
            fig_scores.add_hline(y=0.55, line_dash="dash", line_color="white",
                                 annotation_text="Pass Threshold (0.55)")
            fig_scores.update_layout(
                plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#ffffff",
                xaxis=dict(gridcolor="#2a2d3e", tickangle=-30),
                yaxis=dict(gridcolor="#2a2d3e"),
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=45, b=80)
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        with chart_col2:
            # Radar/Spider chart: Avg scores per metric
            categories = ["Accuracy", "Precision", "Recall", "F1 Score",
                          "Task Success", "Semantic Score"]
            values = [accuracy, precision, recall, f1,
                      task_success_rate / 100, avg_semantic]
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(46,134,222,0.2)',
                line=dict(color='#2e86de', width=2),
                marker=dict(color='#2e86de', size=6)
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2d3e"),
                    angularaxis=dict(gridcolor="#2a2d3e"),
                    bgcolor="#1a1d27"
                ),
                paper_bgcolor="#1a1d27", font_color="#ffffff",
                title=dict(text="Agent Performance Radar", font=dict(color="#ffffff")),
                margin=dict(l=30, r=30, t=50, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            # Score comparison: Semantic vs Keyword vs Final
            fig_multi = go.Figure()
            short_qs = [q[:20] + "..." for q in df_eval["Question"]]
            fig_multi.add_trace(go.Bar(name="Semantic", x=short_qs,
                y=df_eval["Semantic Score"], marker_color="#2e86de"))
            fig_multi.add_trace(go.Bar(name="Keyword", x=short_qs,
                y=df_eval["Keyword Score"], marker_color="#7ed321"))
            fig_multi.add_trace(go.Bar(name="Final", x=short_qs,
                y=df_eval["Final Score"], marker_color="#f39c12"))
            fig_multi.update_layout(
                barmode="group", title="Score Comparison per Question",
                plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#ffffff",
                xaxis=dict(gridcolor="#2a2d3e", tickangle=-30),
                yaxis=dict(gridcolor="#2a2d3e", range=[0, 1]),
                margin=dict(l=20, r=20, t=45, b=80)
            )
            st.plotly_chart(fig_multi, use_container_width=True)

        with chart_col4:
            # Pass/Fail pie
            fig_pf = px.pie(
                values=[pass_count, fail_count],
                names=["Pass ✅", "Fail ❌"],
                color_discrete_sequence=["#7ed321", "#e74c3c"],
                title="Test Pass / Fail Distribution",
                hole=0.5
            )
            fig_pf.update_layout(
                plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27",
                font_color="#ffffff", margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig_pf, use_container_width=True)

        # ── Category-level breakdown ───────────────────────────
        st.markdown("#### 📂 Performance by Query Category")
        df_cat = df_eval.groupby("Category").agg(
            Avg_Final=("Final Score", "mean"),
            Avg_Semantic=("Semantic Score", "mean"),
            Task_Success=("Task Success", "mean"),
            Count=("Question", "count")
        ).reset_index().round(3)
        df_cat.columns = ["Category", "Avg Final Score", "Avg Semantic", "Task Success Rate", "Test Count"]
        st.dataframe(df_cat, use_container_width=True, hide_index=True)

    else:
        st.info("👆 Click **Run Full Evaluation** to evaluate your AI agent against the test suite.")

    st.markdown("---")

    # ════════════════════════════════════
    # LIVE CHAT EVALUATION (from Ask AI tab usage)
    # ════════════════════════════════════
    st.markdown("### 💬 Live Chat Evaluation")
    st.markdown("These scores are collected automatically as you use the **Ask AI** tab.")

    if "live_eval_data" not in st.session_state or not st.session_state["live_eval_data"]:
        st.warning("⚠️ No live data yet. Go to the Ask AI tab and ask some questions first.")
    else:
        df_live = pd.DataFrame(st.session_state["live_eval_data"])
        st.dataframe(df_live, use_container_width=True, hide_index=True)

        avg_final    = df_live["Final Score"].mean()
        avg_semantic = df_live["Semantic Score"].mean()
        live_pass    = sum(1 for r in df_live["Result"] if "Pass" in r)

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.metric("🎯 Avg Final Score", f"{avg_final:.3f}")
        with lc2:
            st.metric("🧠 Avg Semantic Score", f"{avg_semantic:.3f}")
        with lc3:
            st.metric("✅ Questions Passed", f"{live_pass}/{len(df_live)}")

        df_live["Short Q"] = df_live["Question"].str[:25] + "..."
        fig_live = px.bar(df_live, x="Short Q", y="Final Score", color="Final Score",
                          color_continuous_scale=["#e74c3c", "#f39c12", "#7ed321"],
                          title="Live Chat — Final Score per Question", range_y=[0, 1])
        fig_live.add_hline(y=0.55, line_dash="dash", line_color="white",
                           annotation_text="Pass Threshold")
        fig_live.update_layout(
            plot_bgcolor="#1a1d27", paper_bgcolor="#1a1d27", font_color="#ffffff",
            xaxis=dict(gridcolor="#2a2d3e", tickangle=-30),
            yaxis=dict(gridcolor="#2a2d3e"),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=45, b=80)
        )
        st.plotly_chart(fig_live, use_container_width=True)

        if avg_final >= 0.7:
            st.success("✅ Agent performing well on live queries")
        else:
            st.warning("⚠️ Agent needs improvement on live queries")