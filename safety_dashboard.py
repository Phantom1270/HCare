"""
MedVigilant Safety Dashboard
==============================

Interactive Streamlit dashboard to visualize adversarial safety audit results
and run live tests against MedVigilant's AI pipelines.

Run: streamlit run safety_dashboard.py
"""

import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="MedVigilant Safety Audit",
    page_icon="🛡️",
    layout="wide"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
    <style>
    .safety-score-high { color: #00c853; font-size: 3em; font-weight: 800; }
    .safety-score-mid { color: #ffa726; font-size: 3em; font-weight: 800; }
    .safety-score-low { color: #ff1744; font-size: 3em; font-weight: 800; }
    .metric-card {
        background: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .pass-badge { background: #00c853; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
    .fail-badge { background: #ff1744; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
    .partial-badge { background: #ffa726; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("🛡️ MedVigilant Safety Audit Dashboard")
st.markdown("Adversarial robustness evaluation for AI-powered medical safety pipelines.")
st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar: Audit Controls
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.header("⚙️ Audit Controls")
    
    pipeline_filter = st.selectbox(
        "Pipeline to test:",
        ["all", "drug_input", "llm_query", "image_classification", "ocr_pipeline"],
        format_func=lambda x: {
            "all": "🔬 All Pipelines",
            "drug_input": "💊 Drug Input",
            "llm_query": "🤖 LLM Query",
            "image_classification": "🖼️ Image Classification",
            "ocr_pipeline": "📝 OCR Pipeline"
        }.get(x, x)
    )
    
    run_live = st.button("🚀 Run Live Audit", type="primary", use_container_width=True)
    
    st.divider()
    
    st.markdown("### 📂 Load Previous Report")
    report_file = st.file_uploader("Upload safety_report.json", type=["json"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

report = None

# Run live audit
if run_live:
    with st.spinner("Running adversarial safety audit..."):
        from safety.evaluator import SafetyEvaluator
        evaluator = SafetyEvaluator(verbose=False)
        report = evaluator.run_full_audit(pipeline_filter=pipeline_filter)
        st.session_state["last_report"] = report
        evaluator.save_report(report)
        st.success("Audit complete! Results saved to safety_report.json")

# Load from uploaded file
elif report_file:
    report = json.load(report_file)
    st.session_state["last_report"] = report

# Load from session or file
elif "last_report" in st.session_state:
    report = st.session_state["last_report"]
elif os.path.exists("safety_report.json"):
    with open("safety_report.json") as f:
        report = json.load(f)
    st.session_state["last_report"] = report

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Display Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if report is None:
    st.info("👈 Click **Run Live Audit** in the sidebar or upload a previous report to get started.")
    
    # Show what will be tested
    st.markdown("### 🔬 What This Audit Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Attack Vectors
        - 🔐 **NoSQL/SQL Injection** — MongoDB operator injection in drug names
        - 🎭 **Homoglyph Attacks** — Cyrillic characters mimicking Latin letters
        - 💉 **Prompt Injection** — Instruction override, jailbreak, prompt leaking
        - 🖼️ **Adversarial Images** — FGSM perturbations, OOD inputs, noise
        - 📝 **Visual Prompt Injection** — Adversarial text embedded in images
        - ⚠️ **Unsafe Medical Content** — Dangerous dosage, harmful intent
        """)
    
    with col2:
        st.markdown("""
        #### Pipelines Tested
        - 💊 **Drug Input** — RxNorm resolution + medicine cabinet
        - 🤖 **LLM Query** — Gemini-powered medical Q&A
        - 🖼️ **Image Classification** — DermAlert DINOv2 classifier
        - 📝 **OCR Pipeline** — Prescription & label scanners
        
        #### Guardrails Evaluated
        - `sanitize_input()` — Text injection detection
        - `validate_image()` — Image safety validation  
        - `validate_output()` — LLM output safety check
        - `confidence_gate()` — Classification confidence threshold
        """)

else:
    summary = report["summary"]
    score = summary["safety_score"]
    
    # ── Score Display ──
    score_class = "high" if score >= 80 else "mid" if score >= 60 else "low"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="safety-score-{score_class}">{score:.1f}%</div>
            <div style="opacity: 0.7;">Safety Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("✅ Passed", summary["passed"])
    
    with col3:
        st.metric("❌ Failed", summary["failed"])
    
    with col4:
        st.metric("⚠️ Partial", summary["partial"])
    
    st.caption(f"Audit timestamp: {report.get('timestamp', 'Unknown')}")
    st.divider()
    
    # ── Per-Pipeline Breakdown ──
    st.subheader("📊 Per-Pipeline Breakdown")
    
    pipeline_labels = {
        "drug_input": "💊 Drug Input",
        "llm_query": "🤖 LLM Query",
        "image_classification": "🖼️ Image Classification",
        "ocr_pipeline": "📝 OCR Pipeline",
    }
    
    cols = st.columns(len(report.get("by_pipeline", {})))
    
    for i, (pipeline, stats) in enumerate(report.get("by_pipeline", {}).items()):
        with cols[i]:
            label = pipeline_labels.get(pipeline, pipeline)
            total = stats["total"]
            passed = stats["passed"]
            pct = (passed / total * 100) if total > 0 else 0
            
            st.markdown(f"**{label}**")
            st.progress(pct / 100)
            st.caption(f"{passed}/{total} passed ({pct:.0f}%)")
            
            if stats["failed"] > 0:
                st.error(f"{stats['failed']} attacks not caught")
    
    st.divider()
    
    # ── Attack Type Results Table ──
    st.subheader("🎯 Per-Attack-Type Results")
    
    attack_data = []
    for attack_type, verdict in sorted(report.get("by_attack_type", {}).items()):
        icon = "✅" if verdict == "PASS" else "❌" if verdict == "FAIL" else "⚠️"
        attack_data.append({
            "Status": icon,
            "Attack Type": attack_type.replace("_", " ").title(),
            "Verdict": verdict,
        })
    
    if attack_data:
        st.table(attack_data)
    
    st.divider()
    
    # ── Detailed Results ──
    st.subheader("🔍 Detailed Test Results")
    
    # Filter controls
    filter_verdict = st.selectbox(
        "Filter by verdict:",
        ["All", "PASS", "FAIL", "PARTIAL", "ERROR"]
    )
    
    for result in report.get("details", []):
        if filter_verdict != "All" and result.get("verdict") != filter_verdict:
            continue
        
        tc = result.get("test_case", {})
        verdict = result.get("verdict", "UNKNOWN")
        
        badge_class = {
            "PASS": "pass-badge",
            "FAIL": "fail-badge",
            "PARTIAL": "partial-badge",
        }.get(verdict, "partial-badge")
        
        with st.expander(
            f"{'✅' if verdict == 'PASS' else '❌' if verdict == 'FAIL' else '⚠️'} "
            f"[{tc.get('attack_type', '?')}] {tc.get('description', 'No description')[:80]}"
        ):
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.markdown(f"**Input:** `{str(tc.get('input', ''))[:100]}`")
                st.markdown(f"**Attack Type:** {tc.get('attack_type', 'Unknown')}")
                st.markdown(f"**Expected:** {tc.get('expected', 'Unknown')}")
            
            with col_b:
                st.markdown(f"**Verdict:** <span class='{badge_class}'>{verdict}</span>", unsafe_allow_html=True)
                st.markdown(f"**Stage:** {result.get('stage', 'N/A')}")
                
                flags = result.get("actual_flags", [])
                if flags:
                    st.markdown(f"**Flags:** {', '.join(str(f) for f in flags)}")
                
                if result.get("note"):
                    st.info(result["note"])
                
                if result.get("llm_response"):
                    st.markdown(f"**LLM Response:** {result['llm_response'][:150]}...")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("---")
st.markdown("🛡️ MedVigilant Safety Dashboard | Adversarial Robustness Evaluation Framework")
