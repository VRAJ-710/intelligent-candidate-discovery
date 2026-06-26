import os
import streamlit as st
from src.pipeline import run_pipeline

st.set_page_config(page_title="Candidate Ranking Demo",layout="centered")
st.title(" Intelligent Candidate Ranking")
st.write("""
    Upload a small candidate dataset (≤100 candidates).
    Click **Run Ranking** to generate the ranked CSV. """)

uploaded_candidates = st.file_uploader("Upload candidates (.jsonl)",type=["jsonl"])
if uploaded_candidates is not None:
    os.makedirs("temp", exist_ok=True)
    candidate_path = "temp/candidates.jsonl"
    with open(candidate_path, "wb") as f:
        f.write(uploaded_candidates.getbuffer())
    if st.button("Run Ranking"):
        with st.spinner("Ranking candidates..."):
            run_pipeline(candidates_path=candidate_path,output_path="temp/submission.csv",top_k=100,team_id="team_001")
        csv_path = "temp/team_001.csv"
        st.success("Ranking completed!")
        with open(csv_path, "rb") as f:
            st.download_button(
                label="⬇ Download Ranked CSV",
                data=f,
                file_name="team_001.csv",
                mime="text/csv"
            )