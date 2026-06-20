import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────
#  JD-derived constants  (from the actual JD)
# ─────────────────────────────────────────────

# Must-have skills mentioned in JD
CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "vector search", "faiss", "milvus",
    "pinecone", "qdrant", "weaviate", "opensearch", "elasticsearch",
    "information retrieval", "bm25", "retrieval", "ranking", "nlp",
    "fine-tuning llms", "lora", "qlora", "peft", "hugging face transformers",
    "llm", "rag", "langchain", "prompt engineering", "pytorch", "python",
    "recommendation systems", "mlops", "kubeflow", "weights & biases",
    "data science", "machine learning", "deep learning", "transformer",
    "vector database", "hybrid search", "reranking", "embeddings",
    "openSearch", "object detection", "cnn", "gans", "yolo",
    "computer vision", "speech recognition", "image classification",
}

# Nice-to-have skills (bonus points)
BONUS_SKILLS = {
    "learning to rank", "xgboost", "ndcg", "a/b testing", "kafka",
    "spark", "airflow", "dbt", "snowflake", "bigquery", "databricks",
    "docker", "kubernetes", "aws", "gcp", "azure", "fastapi", "flask",
}

# Titles that signal strong fit for this JD
GOOD_TITLES = {
    "machine learning engineer", "ml engineer", "ai engineer",
    "data scientist", "nlp engineer", "research engineer",
    "senior machine learning engineer", "senior ai engineer",
    "backend engineer", "data engineer", "software engineer",
    "search engineer", "ranking engineer", "junior ml engineer",
}

# Titles that are poor fit
BAD_TITLES = {
    "accountant", "hr manager", "marketing manager", "civil engineer",
    "mechanical engineer", "graphic designer", "content writer",
    "sales executive", "customer support", "operations manager",
    "project manager", "business analyst",
}

# Consulting firms JD explicitly penalizes
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "mindtree", "tech mahindra",
}

# Preferred locations per JD
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "india"
}


# ─────────────────────────────────────────────
#  Individual scoring functions
# ─────────────────────────────────────────────

def score_skills(row) -> float:
    """Score based on AI/ML skill match with JD."""
    skills = set(row["skill_names"])
    advanced = set(row["skill_advanced"])

    core_matches = skills & CORE_AI_SKILLS
    bonus_matches = skills & BONUS_SKILLS
    advanced_core = advanced & CORE_AI_SKILLS

    # Base: fraction of core skills matched (capped)
    base = min(len(core_matches) / 6.0, 1.0)

    # Bonus for advanced proficiency in core skills
    advanced_bonus = min(len(advanced_core) * 0.08, 0.3)

    # Small bonus for nice-to-have skills
    bonus = min(len(bonus_matches) * 0.03, 0.15)

    # Endorsement trust multiplier
    endorsements = row.get("total_endorsements", 0)
    trust = min(endorsements / 100.0, 1.0)
    trust_multiplier = 0.85 + (0.15 * trust)

    raw = (base + advanced_bonus + bonus) * trust_multiplier
    return min(raw, 1.0)


def score_title_career(row) -> float:
    """Score based on current title and career trajectory."""
    current = row["current_title"].lower()
    all_titles = row["all_titles"].lower()

    score = 0.0

    # Current title match
    if any(t in current for t in GOOD_TITLES):
        score += 0.6
    elif any(t in current for t in BAD_TITLES):
        score += 0.0
    else:
        score += 0.2  # neutral title

    # Past titles show trajectory toward ML/AI
    good_past = sum(1 for t in GOOD_TITLES if t in all_titles)
    score += min(good_past * 0.1, 0.3)

    # Penalize if only consulting background
    if row["only_consulting"]:
        score *= 0.5

    # Bonus: product company experience (non-consulting)
    all_companies = row.get("all_titles", "").lower()
    if not row["only_consulting"]:
        score += 0.1

    return min(score, 1.0)


def score_experience(row) -> float:
    """Score years of experience against JD requirement (5-9 years ideal)."""
    yoe = row["years_of_experience"]

    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.85
    elif 9 < yoe <= 12:
        return 0.8
    elif 3 <= yoe < 4:
        return 0.65
    elif yoe > 12:
        return 0.65
    elif 2 <= yoe < 3:
        return 0.4
    else:
        return 0.2


def score_education(row) -> float:
    """Score education tier."""
    tier = row["best_edu_tier"]
    mapping = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.45}
    return mapping.get(tier, 0.45)


def score_behavioral(row) -> float:
    """Score behavioral/engagement signals from redrob_signals."""
    score = 0.0
    weights = 0.0

    # 1. Recency — how recently were they active?
    try:
        last_active = datetime.strptime(row["last_active_date"], "%Y-%m-%d")
        days_ago = (datetime.now() - last_active).days
        if days_ago <= 30:
            recency = 1.0
        elif days_ago <= 90:
            recency = 0.8
        elif days_ago <= 180:
            recency = 0.6
        elif days_ago <= 365:
            recency = 0.35
        else:
            recency = 0.1
    except Exception:
        recency = 0.3
    score += recency * 0.25
    weights += 0.25

    # 2. Recruiter response rate
    rr = row.get("recruiter_response_rate", 0)
    score += min(rr, 1.0) * 0.20
    weights += 0.20

    # 3. Profile completeness
    pc = row.get("profile_completeness", 0) / 100.0
    score += pc * 0.15
    weights += 0.15

    # 4. Interview completion rate
    icr = row.get("interview_completion_rate", 0)
    if icr >= 0:
        score += min(icr, 1.0) * 0.15
    weights += 0.15

    # 5. GitHub activity (strong signal for AI engineers)
    gh = row.get("github_activity_score", -1)
    if gh > 0:
        score += min(gh / 100.0, 1.0) * 0.15
    weights += 0.15

    # 6. Notice period (JD prefers sub-30 days)
    np_days = row.get("notice_period_days", 90)
    if np_days <= 30:
        np_score = 1.0
    elif np_days <= 60:
        np_score = 0.75
    elif np_days <= 90:
        np_score = 0.5
    else:
        np_score = 0.25
    score += np_score * 0.10
    weights += 0.10

    return score / weights if weights > 0 else 0.0


def score_location(row) -> float:
    """Score location fit against JD preferred locations."""
    location = (row.get("location", "") + " " + row.get("country", "")).lower()
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return 1.0
    elif row.get("willing_to_relocate", False):
        return 0.7
    else:
        return 0.4


def score_assessment(row) -> float:
    """Score verified skill assessments (objective signal)."""
    assessments = row.get("skill_assessment_scores", {})
    if not assessments:
        return 0.5  # neutral if no assessments taken

    # Only count assessments for AI-relevant skills
    relevant = {
        k: v for k, v in assessments.items()
        if k.lower() in CORE_AI_SKILLS
    }
    if not relevant:
        return 0.5

    avg = sum(relevant.values()) / len(relevant)
    return min(avg / 100.0, 1.0)


# ─────────────────────────────────────────────
#  Master scoring function
# ─────────────────────────────────────────────

WEIGHTS = {
    "skills":      0.30,
    "title":       0.25,
    "experience":  0.15,
    "behavioral":  0.15,
    "location":    0.05,
    "education":   0.05,
    "assessment":  0.05,
}

def score_candidate(row) -> dict:
    """Compute all scores for one candidate and return breakdown."""
    s_skills      = score_skills(row)
    s_title       = score_title_career(row)
    s_experience  = score_experience(row)
    s_behavioral  = score_behavioral(row)
    s_location    = score_location(row)
    s_education   = score_education(row)
    s_assessment  = score_assessment(row)

    final = (
        s_skills     * WEIGHTS["skills"]     +
        s_title      * WEIGHTS["title"]      +
        s_experience * WEIGHTS["experience"] +
        s_behavioral * WEIGHTS["behavioral"] +
        s_location   * WEIGHTS["location"]   +
        s_education  * WEIGHTS["education"]  +
        s_assessment * WEIGHTS["assessment"]
    )

    return {
        "score":            round(final, 4),
        "score_skills":     round(s_skills, 4),
        "score_title":      round(s_title, 4),
        "score_experience": round(s_experience, 4),
        "score_behavioral": round(s_behavioral, 4),
        "score_location":   round(s_location, 4),
        "score_education":  round(s_education, 4),
        "score_assessment": round(s_assessment, 4),
    }


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    """Score all candidates and return DataFrame sorted by score."""
    from src.data_quality import detect_anomalies

    # Run anomaly detection first
    df = detect_anomalies(df)

    print("Scoring all candidates...")

    score_rows = []
    for _, row in df.iterrows():
        scores = score_candidate(row)

        # Apply trust score as multiplier on final score
        trust = row.get("trust_score", 1.0)
        scores["score"] = round(scores["score"] * trust, 4)
        scores["trust_score"] = trust
        scores["anomaly_count"] = row.get("anomaly_count", 0)
        scores["anomaly_flags"] = str(row.get("anomaly_flags", []))

        score_rows.append(scores)

    scores_df = pd.DataFrame(score_rows)
    result = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    result = result.sort_values(
        ["score", "candidate_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    print(f"Scoring complete. Top score: {result['score'].iloc[0]:.4f}")
    return result
