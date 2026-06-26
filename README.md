# Intelligent Candidate Discovery

An AI-powered candidate ranking system that analyzes candidate profiles against a job description and generates a ranked list of the top candidates with explainable reasoning.

---

## Features

- Semantic candidate ranking
- Multi-factor scoring (skills, experience, education, etc.)
- Explainable ranking with candidate-specific reasoning
- Generates submission-ready CSV
- Streamlit demo for reproducibility
- Supports ranking large candidate datasets

---

## Project Structure

```text
.
├── app.py                     # Streamlit demo
├── requirements.txt
├── README.md
├── submission_metadata.yaml
├── data/
│   ├── candidates.jsonl
│   ├── output/
│   └── ...
├── src/
│   ├── pipeline.py
│   ├── data_loader.py
│   ├── scorer.py
│   ├── reasoning.py
│   ├── data_quality.py
│   └── ...
```


## Reproduce Results

Run the complete ranking pipeline:

```bash
python src/pipeline.py
```

The ranked submission CSV will be generated in:

```text
data/output/team_001.csv
```

---

## Streamlit Demo

Run the demo locally:

```bash
streamlit run app.py
```

The application allows users to:

- Upload a candidate JSONL file (≤100 candidates)
- Execute the ranking pipeline
- Download the generated ranked CSV

---

## Output

The generated CSV contains:

- Candidate ID
- Rank
- Final Score
- Ranking Reasoning

---

## Technologies Used

- Python
- Pandas
- NumPy
- Streamlit
- python-docx
- tqdm

---

## AI Assistance

Development was assisted using:

- ChatGPT
- Claude
- GitHub Copilot

AI tools were used for debugging, architecture discussions, documentation, and development assistance. The candidate ranking algorithm and scoring logic are implemented within this repository.

---

## Compute Environment

- CPU-based inference
- Offline ranking
- No external API calls during ranking
- Supports reproducible execution
