"""
Candidate scoring for the Senior AI Engineer role at Redrob AI.
Primary signal: evidence of retrieval, ranking, and search-relevance work in
career descriptions — NOT raw AI keyword counts on skill lists.
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime
# ── Backward-compat exports (used by reasoning.py & data_quality.py) ─────────
CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "vector search", "faiss", "milvus",
    "pinecone", "qdrant", "weaviate", "opensearch", "elasticsearch",
    "information retrieval", "bm25", "retrieval", "ranking", "nlp",
    "fine-tuning llms", "lora", "qlora", "peft", "hugging face transformers",
    "llm", "rag", "langchain", "prompt engineering", "pytorch", "python",
    "recommendation systems", "mlops", "kubeflow", "weights & biases",
    "data science", "machine learning", "deep learning", "transformer",
    "vector database", "hybrid search", "reranking",
    "learning to rank", "ndcg", "mrr", "map",
}
BONUS_SKILLS = {
    "xgboost", "a/b testing", "kafka", "spark", "airflow", "dbt",
    "snowflake", "bigquery", "databricks", "docker", "kubernetes",
    "aws", "gcp", "azure", "fastapi", "flask",
}
GOOD_TITLES = {
    "machine learning engineer", "ml engineer", "ai engineer",
    "data scientist", "nlp engineer", "research engineer",
    "senior machine learning engineer", "senior ai engineer",
    "search engineer", "ranking engineer", "recommendation engineer",
    "backend engineer", "data engineer", "software engineer",
}
BAD_TITLES = {
    "accountant", "hr manager", "marketing manager", "civil engineer",
    "mechanical engineer", "graphic designer", "content writer",
    "sales executive", "customer support", "operations manager",
    "project manager", "business analyst",
}
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "mindtree", "tech mahindra",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "gurugram", "india",
}
# ── Career-text keyword groups (strongest JD signals) ────────────────────────

RETRIEVAL_SEARCH_KEYWORDS = (
    "information retrieval", "retrieval system", "retrieval pipeline",
    "search system", "search engine", "search infrastructure",
    "search relevance", "semantic search", "vector search", "vector database",
    "vector db", "embedding", "embeddings", "dense retrieval", "sparse retrieval",
    "hybrid search", "bm25", "passage retrieval", "document retrieval",
    "faiss", "milvus", "pinecone", "qdrant", "weaviate", "opensearch",
    "elasticsearch", "approximate nearest neighbor", "ann index",
    "inverted index", "query understanding", "indexing pipeline",
)

RANKING_RECOMMENDATION_KEYWORDS = (
    "learning to rank", "learning-to-rank", "ltr model", "ltr pipeline",
    "ranking model", "ranking system", "ranking pipeline", "reranking",
    "re-ranking", "relevance ranking", "search ranking", "candidate ranking",
    "recommendation system", "recommender system", "recommender engine",
    "personalization engine", "collaborative filtering", "content-based filtering",
    "feed ranking", "two-tower", "two tower", "click-through rate",
    "conversion ranking",
)

PRODUCTION_ML_KEYWORDS = (
    "production ml", "production model", "production deployment",
    "model serving", "model deployment", "serving infrastructure",
    "ml pipeline", "ml platform", "mlops", "inference at scale",
    "real-time inference", "online inference", "batch inference",
    "feature store", "model monitoring", "model registry",
    "a/b test", "ab test", "experiment platform", "online serving",
    "low-latency inference", "scalable ml", "end-to-end ml",
)

EVALUATION_METRICS_KEYWORDS = (
    "ndcg", "mrr", "map@", "map score", "mean average precision",
    "precision@", "recall@", "hit rate", "offline evaluation",
    "online evaluation", "ranking metrics", "search evaluation",
    "relevance evaluation", "evaluation framework", "benchmark suite",
    "human evaluation", "interleaving", "counterfactual evaluation",
)
# ── Scoring weights (retrieval/ranking is dominant) ──────────────────────────
WEIGHTS = {
    "retrieval":     0.34,   # career evidence of IR / search / ranking work
    "stability":     0.16,   # product co. background, tenure, title trajectory
    "recruiter":     0.14,   # platform engagement from recruiters
    "availability":  0.12,   # notice period, activity, open-to-work
    "verification":  0.08,   # assessments + identity verification
    "experience":    0.08,   # YoE band fit (5–9 ideal)
    "skills":        0.04,   # deliberately low — skills list alone is weak
    "location":      0.02,
    "education":     0.02,
}
# Per-flag trust reduction applied inside score_all (on top of trust_score)
ANOMALY_PENALTY_PER_FLAG = 0.025
ANOMALY_PENALTY_FLOOR = 0.70
# ── Helpers ──────────────────────────────────────────────────────────────────
def _career_text(row) -> str:
    """Career descriptions — highest-trust signal for project experience."""
    return (row.get("all_descriptions") or "").lower()


def _profile_text(row) -> str:
    """Headline, summary, titles, and career text combined."""
    return (row.get("full_text") or "").lower()


def _count_hits(text: str, keywords: tuple[str, ...]) -> int:
    if not text:
        return 0
    return sum(1 for kw in keywords if kw in text)
def _group_score(hit_count: int, cap: int = 4) -> float:
    """Diminishing returns: first few keyword hits matter most."""
    return min(hit_count / cap, 1.0)
def _keyword_group_scores(text: str) -> dict[str, float]:
    return {
        "retrieval":  _group_score(_count_hits(text, RETRIEVAL_SEARCH_KEYWORDS)),
        "ranking":    _group_score(_count_hits(text, RANKING_RECOMMENDATION_KEYWORDS)),
        "production": _group_score(_count_hits(text, PRODUCTION_ML_KEYWORDS)),
        "evaluation": _group_score(_count_hits(text, EVALUATION_METRICS_KEYWORDS)),
    }
def _combine_group_scores(groups: dict[str, float]) -> float:
    """
    Weighted blend of keyword-group scores.
    Retrieval + ranking dominate; production + evaluation confirm depth.
    """
    base = (
        groups["retrieval"]  * 0.38 +
        groups["ranking"]    * 0.32 +
        groups["production"] * 0.18 +
        groups["evaluation"] * 0.12
    )
    # Bonus when evidence spans multiple domains (real systems work)
    active_groups = sum(1 for v in groups.values() if v >= 0.25)
    breadth_bonus = {0: 0.0, 1: 0.0, 2: 0.06, 3: 0.12, 4: 0.18}.get(active_groups, 0.18)
    return min(base + breadth_bonus, 1.0)
# ── New primary scoring functions ────────────────────────────────────────────

def score_retrieval_ranking(row) -> float:
    """
    Score evidence of retrieval, ranking, recommendation, and search-relevance
    work from career descriptions and project text — NOT skill tags alone.
    """
    career = _career_text(row)
    profile = _profile_text(row)

    # Career descriptions carry 75 % of the signal; summary/headline 25 %
    career_groups = _keyword_group_scores(career)
    profile_groups = _keyword_group_scores(profile)

    blended_groups = {
        k: career_groups[k] * 0.75 + profile_groups[k] * 0.25
        for k in career_groups
    }
    score = _combine_group_scores(blended_groups)

    # Small supplementary boost when structured skills corroborate career text
    skills = set(row.get("skill_names") or [])
    corroborating = skills & {
        "information retrieval", "vector search", "learning to rank",
        "recommendation systems", "embeddings", "faiss", "milvus",
        "elasticsearch", "opensearch", "ranking", "retrieval", "ndcg",
    }
    if corroborating and score >= 0.20:
        score = min(score + len(corroborating) * 0.02, 1.0)

    # Penalise keyword-stuffing: many AI skills listed but zero career evidence
    career_total_hits = sum(_count_hits(career, g) for g in (
        RETRIEVAL_SEARCH_KEYWORDS, RANKING_RECOMMENDATION_KEYWORDS,
        PRODUCTION_ML_KEYWORDS, EVALUATION_METRICS_KEYWORDS,
    ))
    skill_ai_count = len(skills & CORE_AI_SKILLS)
    if skill_ai_count >= 6 and career_total_hits == 0:
        score *= 0.40
    elif skill_ai_count >= 4 and career_total_hits <= 1:
        score *= 0.60

    # Title boost for search/ranking engineers with real career evidence
    current = (row.get("current_title") or "").lower()
    search_titles = {"search engineer", "ranking engineer", "recommendation engineer",
                     "ml engineer", "machine learning engineer", "ai engineer"}
    if any(t in current for t in search_titles) and career_total_hits >= 2:
        score = min(score + 0.08, 1.0)

    return min(score, 1.0)
def score_recruiter_interest(row) -> float:
    """Recruiter-side engagement: saves, views, response rate, open-to-work."""
    score = 0.0

    rr = min(row.get("recruiter_response_rate", 0), 1.0)
    score += rr * 0.35

    saves = row.get("saved_by_recruiters_30d", 0)
    save_score = min(saves / 15.0, 1.0)
    score += save_score * 0.30

    views = row.get("profile_views_30d", 0)
    view_score = min(views / 50.0, 1.0)
    score += view_score * 0.20

    if row.get("open_to_work", False):
        score += 0.10
    # Fast response time is a positive recruiter signal
    avg_hours = row.get("avg_response_time_hours", 999)
    if avg_hours <= 24:
        score += 0.05
    elif avg_hours <= 72:
        score += 0.03

    return min(score, 1.0)
def score_stability(row) -> float:
    """
    Career stability and company-type fit.
    Product-company experience preferred; pure consulting penalised.
    """
    score = 0.0

    current = (row.get("current_title") or "").lower()
    all_titles = (row.get("all_titles") or "").lower()

    if any(t in current for t in GOOD_TITLES):
        score += 0.35
    elif any(t in current for t in BAD_TITLES):
        score += 0.05
    else:
        score += 0.18

    good_past = sum(1 for t in GOOD_TITLES if t in all_titles)
    score += min(good_past * 0.06, 0.18)

    # Product-company vs consulting
    if row.get("only_consulting", False):
        score *= 0.45
    else:
        score += 0.25

    # Tenure stability from career_history durations
    history = row.get("career_history") or []
    if history:
        durations = [j.get("duration_months", 0) for j in history if j.get("duration_months")]
        if durations:
            median_months = sorted(durations)[len(durations) // 2]
            if median_months >= 30:
                score += 0.15
            elif median_months >= 18:
                score += 0.10
            elif median_months >= 12:
                score += 0.05
            else:
                score += 0.0  # frequent job changes
    return min(score, 1.0)
def score_verification(row) -> float:
    """Objective verification: assessments, identity checks, GitHub activity."""
    score = 0.0
    weights = 0.0

    assessments = row.get("skill_assessment_scores") or {}
    if assessments:
        retrieval_skills = {
            k: v for k, v in assessments.items()
            if any(kw in k.lower() for kw in (
                "retrieval", "ranking", "search", "ndcg", "ml", "python",
                "embedding", "recommendation",
            ))
        }
        pool = retrieval_skills or assessments
        avg = sum(pool.values()) / len(pool)
        score += min(avg / 100.0, 1.0) * 0.45
        weights += 0.45

    gh = row.get("github_activity_score", -1)
    if gh > 0:
        score += min(gh / 100.0, 1.0) * 0.25
    weights += 0.25
    identity_checks = [
        row.get("verified_email", False),
        row.get("verified_phone", False),
        row.get("linkedin_connected", False),
    ]
    score += sum(identity_checks) / 3.0 * 0.20
    weights += 0.20

    pc = row.get("profile_completeness", 0) / 100.0
    score += pc * 0.10
    weights += 0.10

    return score / weights if weights > 0 else 0.35

def score_availability(row) -> float:
    """Hiring readiness: notice period, recent activity, relocation willingness."""
    score = 0.0
    np_days = row.get("notice_period_days", 90)
    if np_days <= 30:
        score += 0.35
    elif np_days <= 60:
        score += 0.25
    elif np_days <= 90:
        score += 0.15
    else:
        score += 0.05

    try:
        last_active = datetime.strptime(row["last_active_date"], "%Y-%m-%d")
        days_ago = (datetime.now() - last_active).days
        if days_ago <= 30:
            score += 0.30
        elif days_ago <= 90:
            score += 0.22
        elif days_ago <= 180:
            score += 0.14
        elif days_ago <= 365:
            score += 0.06
    except (TypeError, ValueError):
        score += 0.10

    if row.get("open_to_work", False):
        score += 0.20

    icr = row.get("interview_completion_rate", 0)
    if icr >= 0:
        score += min(icr, 1.0) * 0.10

    location = (row.get("location", "") + " " + row.get("country", "")).lower()
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        score += 0.05
    elif row.get("willing_to_relocate", False):
        score += 0.03

    return min(score, 1.0)
# ── Supporting scoring functions (reduced weight) ────────────────────────────

def score_skills(row) -> float:
    """Lightweight skill-list check — supplementary only, not primary."""
    skills = set(row.get("skill_names") or [])
    advanced = set(row.get("skill_advanced") or [])

    retrieval_skills = skills & {
        "information retrieval", "vector search", "learning to rank",
        "recommendation systems", "embeddings", "faiss", "milvus",
        "elasticsearch", "opensearch", "ranking", "retrieval", "ndcg",
        "hybrid search", "reranking", "bm25",
    }
    base = min(len(retrieval_skills) / 4.0, 0.70)

    advanced_retrieval = advanced & retrieval_skills
    base += min(len(advanced_retrieval) * 0.05, 0.20)

    bonus = min(len(skills & BONUS_SKILLS) * 0.02, 0.10)
    return min(base + bonus, 1.0)
def score_experience(row) -> float:
    """YoE band fit — 5–9 years ideal for Senior AI Engineer."""
    yoe = row.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        return 1.0
    if 4 <= yoe < 5:
        return 0.85
    if 9 < yoe <= 12:
        return 0.80
    if 3 <= yoe < 4:
        return 0.65
    if yoe > 12:
        return 0.60
    if 2 <= yoe < 3:
        return 0.40
    return 0.20
def score_education(row) -> float:
    tier = row.get("best_edu_tier", 4)
    return {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.45}.get(tier, 0.45)
def score_location(row) -> float:
    location = (row.get("location", "") + " " + row.get("country", "")).lower()
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return 1.0
    if row.get("willing_to_relocate", False):
        return 0.70
    return 0.40
# ── Master scoring ───────────────────────────────────────────────────────────

def score_candidate(row) -> dict:
    """Compute all component scores and return breakdown dict."""
    s_retrieval    = score_retrieval_ranking(row)
    s_recruiter    = score_recruiter_interest(row)
    s_stability    = score_stability(row)
    s_verification = score_verification(row)
    s_availability = score_availability(row)
    s_skills       = score_skills(row)
    s_experience   = score_experience(row)
    s_location     = score_location(row)
    s_education    = score_education(row)

    final = (
        s_retrieval    * WEIGHTS["retrieval"]    +
        s_stability    * WEIGHTS["stability"]    +
        s_recruiter    * WEIGHTS["recruiter"]    +
        s_availability * WEIGHTS["availability"] +
        s_verification * WEIGHTS["verification"] +
        s_experience   * WEIGHTS["experience"]   +
        s_skills       * WEIGHTS["skills"]       +
        s_location     * WEIGHTS["location"]     +
        s_education    * WEIGHTS["education"]
    )

    return {
        "score":              round(final, 4),
        "score_retrieval":    round(s_retrieval, 4),
        "score_recruiter":    round(s_recruiter, 4),
        "score_stability":    round(s_stability, 4),
        "score_verification": round(s_verification, 4),
        "score_availability": round(s_availability, 4),
        "score_skills":       round(s_skills, 4),
        "score_experience":   round(s_experience, 4),
        "score_location":     round(s_location, 4),
        "score_education":    round(s_education, 4),
    }


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    """Score all candidates and return DataFrame sorted by score descending."""
    from src.data_quality import detect_anomalies

    df = detect_anomalies(df)
    print("Scoring all candidates...")

    score_rows = []
    for _, row in df.iterrows():
        scores = score_candidate(row)

        trust = row.get("trust_score", 1.0)
        anomaly_count = row.get("anomaly_count", 0)
        anomaly_multiplier = max(
            1.0 - anomaly_count * ANOMALY_PENALTY_PER_FLAG,
            ANOMALY_PENALTY_FLOOR,
        )

        scores["score"] = round(scores["score"] * trust * anomaly_multiplier, 4)
        scores["trust_score"] = trust
        scores["anomaly_count"] = anomaly_count
        scores["anomaly_flags"] = str(row.get("anomaly_flags", []))

        score_rows.append(scores)

    scores_df = pd.DataFrame(score_rows)
    result = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    result = result.sort_values(
        ["score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    print(f"Scoring complete. Top score: {result['score'].iloc[0]:.4f}")
    return result
