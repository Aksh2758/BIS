import streamlit as st
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import BISRagPipeline

st.set_page_config(
    page_title="BIS Smart Compliance AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root theme ── */
:root {
    --navy:     #0F1419;
    --navy-2:   #151B26;
    --navy-3:   #1E2839;
    --accent:   #3B82F6;
    --accent-2: #60A5FA;
    --accent-3: #2563EB;
    --green:    #10B981;
    --text-1:   #F8FAFC;
    --text-2:   #A1A9B8;
    --text-3:   #697684;
    --border:   rgba(255,255,255,0.06);
}

/* ── Base reset ── */
.stApp { background: var(--navy) !important; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text-1); }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0F1419 0%, #151B26 50%, #0D1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 5rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -120px; right: -120px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    color: var(--accent-2);
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 0.12em;
    padding: 4px 10px; border-radius: 99px;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.4rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.025em;
    margin: 0 0 0.4rem;
    background: linear-gradient(135deg, #F8FAFC 30%, #A1A9B8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-title span {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 0.9rem; color: var(--text-2);
    max-width: 550px; line-height: 1.5;
    margin: 0;
}
.hero-stats {
    display: flex; gap: 2rem; margin-top: 1.2rem;
}
.stat { text-align: left; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    color: var(--accent);
}
.stat-label { font-size: 10px; color: var(--text-3); letter-spacing: 0.08em; margin-top: 2px; text-transform: uppercase; }

/* ── Search zone ── */
.search-zone {
    padding: 1.8rem 5rem 2rem;
    background: var(--navy-2);
    border-bottom: 1px solid var(--border);
}
.search-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 0.16em;
    color: var(--text-3); text-transform: uppercase;
    margin-bottom: 0.7rem;
    font-weight: 500;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: var(--navy-3) !important;
    border: 1px solid rgba(59,130,246,0.15) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.99rem !important;
    padding: 16px 18px !important;
    resize: none !important;
    transition: all 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-3) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(59,130,246,0.35) !important;
}

/* ── Results area ── */
.results-zone { padding: 2rem 5rem; }
.results-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 1.5rem;
}
.results-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: var(--text-1);
}
.results-count {
    font-family: 'DM Mono', monospace;
    font-size: 11px; color: var(--text-3);
    background: var(--navy-3);
    border: 1px solid var(--border);
    padding: 4px 12px; border-radius: 99px;
}
.latency-pill {
    font-family: 'DM Mono', monospace;
    font-size: 11px; color: var(--green);
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    padding: 4px 12px; border-radius: 99px;
    margin-left: auto;
}

/* ── Standard card ── */
.std-card {
    background: var(--navy-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: all 0.2s;
}
.std-card:hover {
    border-color: rgba(59,130,246,0.2);
    transform: translateX(3px);
    background: rgba(59,130,246,0.03);
}
.std-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
    border-radius: 3px 0 0 3px;
}
.std-card.rank-1::before { background: linear-gradient(180deg, #3B82F6, #60A5FA); }
.std-card.rank-2::before { background: linear-gradient(180deg, #0EA5E9, #06B6D4); }
.std-card.rank-3::before { background: linear-gradient(180deg, #8B5CF6, #A78BFA); }

.card-top { display: flex; align-items: flex-start; gap: 12px; }
.rank-badge {
    font-family: 'DM Mono', monospace;
    font-size: 10px; font-weight: 600;
    color: var(--accent);
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    padding: 4px 8px; border-radius: 6px;
    white-space: nowrap; margin-top: 2px;
    min-width: 30px; text-align: center;
}
.std-label {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: var(--text-1); margin-bottom: 4px;
}
.std-rationale {
    font-size: 0.85rem; color: var(--text-2);
    line-height: 1.5;
}
.std-meta {
    display: flex; gap: 6px; flex-wrap: wrap;
    margin-top: 8px;
}
.meta-tag {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: 0.05em;
    padding: 3px 8px; border-radius: 99px;
}
.tag-cement   { background: rgba(59,130,246,0.1);  color: #60A5FA; border: 1px solid rgba(59,130,246,0.2); }
.tag-concrete { background: rgba(99,102,241,0.1);  color: #818CF8; border: 1px solid rgba(99,102,241,0.2); }
.tag-steel    { background: rgba(6,182,212,0.1);   color: #06B6D4; border: 1px solid rgba(6,182,212,0.2); }
.tag-masonry  { background: rgba(139,92,246,0.1);  color: #A78BFA; border: 1px solid rgba(139,92,246,0.2); }
.tag-aggregates { background: rgba(34,197,94,0.1); color: #86EFAC; border: 1px solid rgba(34,197,94,0.2); }
.tag-pipes    { background: rgba(59,130,246,0.1);  color: #60A5FA; border: 1px solid rgba(59,130,246,0.2); }
.tag-general  { background: rgba(100,116,139,0.1); color: #CBD5E1; border: 1px solid rgba(100,116,139,0.2); }

/* ── Example queries ── */
.examples-row {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-top: 0.8rem;
}
.example-chip {
    font-size: 12px; color: var(--text-2);
    background: var(--navy-3);
    border: 1px solid rgba(59,130,246,0.15);
    padding: 6px 12px; border-radius: 99px;
    cursor: pointer;
    transition: all 0.2s;
    display: inline-block;
    font-weight: 500;
}
.example-chip:hover {
    border-color: var(--accent);
    color: var(--accent-2);
    background: rgba(59,130,246,0.06);
    transform: translateY(-1px);
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3rem 2rem;
    color: var(--text-3);
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-text { font-size: 0.95rem; color: var(--text-2); }

/* ── Metrics row ── */
.metrics-row {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 12px; margin-bottom: 1.8rem;
}
.metric-box {
    background: var(--navy-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem; font-weight: 700;
    color: var(--accent);
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px; color: var(--text-3);
    letter-spacing: 0.08em; margin-top: 2px; text-transform: uppercase;
}

/* Spinner color */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    import json
    try:
        with open("data/standards_chunks_enriched.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
    except FileNotFoundError:
        with open("data/standards_chunks.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
    return BISRagPipeline(docs)


def get_category_tag(metadata):
    cats = metadata.get("categories", ["general"])
    cat = cats[0] if cats else "general"
    labels = {
        "cement": "CEMENT", "concrete": "CONCRETE", "steel": "STEEL",
        "masonry": "MASONRY", "aggregates": "AGGREGATES",
        "pipes": "PIPES", "general": "GENERAL"
    }
    return cat, labels.get(cat, "GENERAL")


def get_type_tags(metadata):
    tags = []
    if metadata.get("is_lightweight"):     tags.append("LIGHTWEIGHT")
    if metadata.get("is_slag_cement"):     tags.append("SLAG CEMENT")
    if metadata.get("is_pozzolana"):       tags.append("POZZOLANA")
    if metadata.get("is_rapid_hardening"): tags.append("RAPID HARDENING")
    if metadata.get("is_white_cement"):    tags.append("WHITE CEMENT")
    if metadata.get("part"):               tags.append(f"PART {metadata['part']}")
    return tags


st.markdown("""
<div class="hero">
    <div class="hero-badge">🏛️ &nbsp; BIS SP 21 · Building Materials</div>
    <h1 class="hero-title">BIS Smart<br><span>Compliance AI</span></h1>
    <p class="hero-sub">Describe your product. Get the exact BIS standards you need in seconds — built for Indian MSEs.</p>
    <div class="hero-stats">
        <div class="stat">
            <div class="stat-num">564</div>
            <div class="stat-label">STANDARDS INDEXED</div>
        </div>
        <div class="stat">
            <div class="stat-num">100%</div>
            <div class="stat-label">HIT RATE @3</div>
        </div>
        <div class="stat">
            <div class="stat-num">&lt;1s</div>
            <div class="stat-label">AVG RESPONSE</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# st.markdown('<div class="search-zone">', unsafe_allow_html=True)
st.markdown('<div class="search-label">Describe your product or manufacturing process</div>', unsafe_allow_html=True)

EXAMPLES = [
    "We manufacture 33 Grade Ordinary Portland Cement",
    "We produce hollow lightweight concrete blocks",
    "Our factory makes UPVC pipes for drainage",
]

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area(
        label="query",
        value=st.session_state.query_text,
        placeholder="e.g. We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement...",
        height=90,
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    search_clicked = st.button("Find Standards", use_container_width=True)

# Example chips
st.markdown('<div class="search-label" style="margin-top:1.5rem">Try an example</div>', unsafe_allow_html=True)
ecols = st.columns(len(EXAMPLES))
for i, (ecol, example) in enumerate(zip(ecols, EXAMPLES)):
    with ecol:
        if st.button(example[:35] + "…" if len(example) > 35 else example,
                     key=f"ex_{i}", use_container_width=True):
            st.session_state.query_text = example
            st.session_state.trigger_search = True
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="results-zone">', unsafe_allow_html=True)

# Trigger search if example was clicked or Find Standards button was clicked
should_search = search_clicked or st.session_state.get("trigger_search", False)

if should_search and query.strip():
    with st.spinner("Searching 564 standards..."):
        pipeline = load_pipeline()
        retrieved, rationale_map, latency = pipeline.query(query.strip(), top_k=5)

    # Clear the trigger flag
    st.session_state.trigger_search = False

    # Load metadata for tags
    try:
        with open("data/standards_chunks_enriched.json", encoding="utf-8") as f:
            all_docs = json.load(f)
        meta_lookup = {d["label"]: d.get("metadata", {}) for d in all_docs}
    except Exception:
        meta_lookup = {}

    # Metrics row
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-box">
            <div class="metric-val">{len(retrieved)}</div>
            <div class="metric-lbl">STANDARDS FOUND</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{latency:.2f}s</div>
            <div class="metric-lbl">RESPONSE TIME</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">RAG</div>
            <div class="metric-lbl">GROQ · LLAMA 3.1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Results header
    st.markdown(f"""
    <div class="results-header">
        <div class="results-title">Top BIS Standards</div>
        <div class="results-count">{len(retrieved)} results</div>
        <div class="latency-pill">⚡ {latency:.2f}s</div>
    </div>
    """, unsafe_allow_html=True)

    rank_classes = ["rank-1", "rank-2", "rank-3", "", ""]
    rank_labels  = ["#1", "#2", "#3", "#4", "#5"]

    for i, label in enumerate(retrieved):
        meta = meta_lookup.get(label, {})
        rationale = rationale_map.get(label) or "Matched based on semantic similarity to product description."
        cat_key, cat_text = get_category_tag(meta)
        type_tags = get_type_tags(meta)

        # Build meta tags HTML
        tags_html = f'<span class="meta-tag tag-{cat_key}">{cat_text}</span>'
        for t in type_tags:
            tags_html += f'<span class="meta-tag tag-general">{t}</span>'

        st.markdown(f"""
        <div class="std-card {rank_classes[i]}">
            <div class="card-top">
                <div class="rank-badge">{rank_labels[i]}</div>
                <div style="flex:1">
                    <div class="std-label">{label}</div>
                    <div class="std-rationale">{rationale}</div>
                    <div class="std-meta">{tags_html}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif search_clicked and not query.strip():
    st.warning("Please enter a product description first.")
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-text">Enter a product description above to find relevant BIS standards</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
