import os
import tempfile
import streamlit as st
import pandas as pd


from main import (
    build_graph,
    score_vendors,
    rank_compare_vendors,
    DEFAULT_SCORING_WEIGHTS,
)

st.set_page_config(page_title="Vendor Evaluation Agent", layout="wide")
st.title("Vendor Evaluation Agent")
st.caption(
    "Upload vendor quotations, adjust scoring weights, and compare vendors side by side.")

if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = []
if "clarification_emails" not in st.session_state:
    st.session_state.clarification_emails = []
if "missing_info" not in st.session_state:
    st.session_state.missing_info = []

# ---------------------------------------------
# STEP 1 — Upload vendor documents
# ---------------------------------------------
st.header("1. Upload vendor quotations")
uploaded_files = st.file_uploader(
    "Upload one PDF per vendor", type="pdf", accept_multiple_files=True
)

vendor_docs = []
if uploaded_files:
    for file in uploaded_files:
        default_name = os.path.splitext(file.name)[0]
        vendor_name = st.text_input(
            f"Vendor name for '{file.name}'", value=default_name, key=file.name)
        tmp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(tmp_path, "wb") as f:
            f.write(file.getbuffer())
        vendor_docs.append({"vendor_name": vendor_name, "file_path": tmp_path})

run_clicked = st.button("Run evaluation", disabled=not vendor_docs)

if run_clicked:
    with st.spinner("Reading documents and extracting vendor data..."):
        app = build_graph()
        result = app.invoke({
            "vendor_docs": vendor_docs,
            "scoring_weights": DEFAULT_SCORING_WEIGHTS,
        })
    st.session_state.extracted_info = result.get("extracted_info", [])
    st.session_state.clarification_emails = result.get(
        "clarification_emails", [])
    st.session_state.missing_info = result.get("missing_info", [])

# ---------------------------------------------
# STEP 2 — Adjust scoring weights (live, no re-extraction)
# ---------------------------------------------
if st.session_state.extracted_info:
    st.header("2. Adjust scoring weights")
    col1, col2, col3, col4 = st.columns(4)
    price_w = col1.slider(
        "Price", 0.0, 1.0, DEFAULT_SCORING_WEIGHTS["price_per_unit"], 0.05)
    lead_w = col2.slider("Lead time", 0.0, 1.0,
                         DEFAULT_SCORING_WEIGHTS["lead_time_days"], 0.05)
    moq_w = col3.slider("MOQ", 0.0, 1.0, DEFAULT_SCORING_WEIGHTS["moq"], 0.05)
    quality_w = col4.slider("Quality certs", 0.0, 1.0,
                            DEFAULT_SCORING_WEIGHTS["quality_certifications"], 0.05)

    total_w = price_w + lead_w + moq_w + quality_w
    weights = (
        {"price_per_unit": price_w / total_w, "lead_time_days": lead_w / total_w,
         "moq": moq_w / total_w, "quality_certifications": quality_w / total_w}
        if total_w > 0 else DEFAULT_SCORING_WEIGHTS
    )
    st.caption(
        f"Weights auto-normalized to sum to 1.0 (raw total: {total_w:.2f})")

    scored = score_vendors(
        {"extracted_info": st.session_state.extracted_info, "scoring_weights": weights})
    ranked = rank_compare_vendors(scored)

    # ---------------------------------------------
    # STEP 3 — Results
    # ---------------------------------------------
    st.header("3. Ranked comparison")
    ranked_vendors = ranked.get("ranked_vendors", [])
    if ranked_vendors:
        df = pd.DataFrame([
            {
                "Rank": v["rank"],
                "Vendor": v["vendor_name"],
                "Score": v.get("score", 0),
                "Price/unit": v.get("price_per_unit"),
                "Currency": v.get("currency"),
                "MOQ": v.get("moq"),
                "Lead time (days)": v.get("lead_time_days"),
                "Payment terms": v.get("payment_terms"),
                "Certifications": v.get("quality_certifications") or "None stated",
            }
            for v in ranked_vendors
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No vendors could be scored yet.")

    if st.session_state.missing_info:
        st.header("Incomplete vendor data")
        for issue in st.session_state.missing_info:
            st.warning(
                f"{issue.get('vendor_name', 'Unknown vendor')}: {issue}")

    if st.session_state.clarification_emails:
        st.header("Drafted clarification emails")
        for email in st.session_state.clarification_emails:
            with st.expander(f"To: {email['vendor_name']} — {email['subject']}"):
                st.write(email["body"])
