import pandas as pd
import numpy as np
from datetime import datetime


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and flag data quality issues in candidate profiles.
    Adds anomaly columns and a trust_score to each candidate.
    Does NOT remove candidates — just flags and adjusts scoring.
    """

    print("Running data quality checks...")
    df = df.copy()

    # Initialize flags
    df["anomaly_flags"] = [[] for _ in range(len(df))]
    df["anomaly_count"] = 0
    df["trust_score"] = 1.0  # starts at full trust, gets reduced

    TODAY = datetime.now()

    for idx, row in df.iterrows():
        flags = []

        # ── 1. Experience vs Age impossibility ──────────────────
        yoe = row.get("years_of_experience", 0)

        # Minimum working age is 18, so max possible experience
        # If someone graduated around 22, max real exp = current_year - 22
        # We don't have DOB but we can infer from education
        edu_end_years = []
        # We stored best_edu_tier but not grad year — use career start
        # Check if YOE is suspiciously high (>35 years is unrealistic)
        if yoe > 35:
            flags.append("impossible_experience_35+")

        # ── 2. Career timeline overlaps ─────────────────────────
        # Check if claimed YOE matches sum of career durations
        # Get total career months from all_descriptions proxy
        # (We don't store individual durations but can check via full_text)
        # We'll check claimed YOE vs realistic max
        if yoe > 0:
            # If current title is entry-level but YOE > 15, suspicious
            entry_titles = {"intern", "trainee", "fresher", "junior", "associate"}
            current = row.get("current_title", "").lower()
            if any(t in current for t in entry_titles) and yoe > 8:
                flags.append("entry_title_high_experience")

        # ── 3. Skill inflation ───────────────────────────────────
        skill_count = row.get("skill_count", 0)
        advanced_count = row.get("advanced_skill_count", 0)

        # More than 15 advanced skills is suspicious
        if advanced_count > 15:
            flags.append("skill_inflation_advanced")

        # Advanced in >80% of all skills is suspicious
        if skill_count > 0 and advanced_count / skill_count > 0.8 and skill_count > 5:
            flags.append("skill_inflation_ratio")

        # ── 4. Endorsement vs Connection anomaly ────────────────
        connections = row.get("connection_count", 0)
        endorsements = row.get("total_endorsements", 0)

        # Having 500 endorsements with 10 connections is impossible
        if connections < 20 and endorsements > 200:
            flags.append("endorsement_connection_mismatch")

        # ── 5. Behavioral date anomalies ────────────────────────
        try:
            last_active = datetime.strptime(
                row["last_active_date"], "%Y-%m-%d"
            )
            if last_active > TODAY:
                flags.append("future_last_active_date")

            signup_str = row.get("signup_date", "")
            if signup_str:
                signup = datetime.strptime(signup_str, "%Y-%m-%d")
                if signup > TODAY:
                    flags.append("future_signup_date")
                if last_active < signup:
                    flags.append("active_before_signup")
        except Exception:
            flags.append("invalid_date_format")

        # ── 6. Assessment score anomalies ───────────────────────
        assessments = row.get("skill_assessment_scores", {})
        if assessments:
            # Perfect 100 on all assessments is suspicious
            scores = list(assessments.values())
            if all(s >= 99 for s in scores) and len(scores) >= 3:
                flags.append("perfect_assessment_scores")

            # Claimed advanced but scored <30 on assessment
            advanced_skills = set(row.get("skill_advanced", []))
            for skill, score in assessments.items():
                if skill.lower() in advanced_skills and score < 30:
                    flags.append(f"advanced_claim_low_assessment:{skill}")

        # ── 7. Title vs Skill mismatch ───────────────────────────
        from src.scorer import CORE_AI_SKILLS, BAD_TITLES
        current_title = row.get("current_title", "").lower()
        skill_names = set(row.get("skill_names", []))
        matched_ai = skill_names & CORE_AI_SKILLS

        # Claims non-tech title but has tons of advanced AI skills
        is_bad_title = any(t in current_title for t in BAD_TITLES)
        if is_bad_title and len(matched_ai) > 8:
            flags.append("title_skill_mismatch")

        # ── 8. Salary anomalies ──────────────────────────────────
        sal_min = row.get("sal_min", 0)
        sal_max = row.get("sal_max", 0)

        # After our fix in data_loader, min should <= max
        # But check if salary is 0 for experienced candidate
        if yoe > 5 and sal_min == 0 and sal_max == 0:
            flags.append("missing_salary_experienced")

        # Unrealistically high salary expectation
        if sal_max > 500:
            flags.append("unrealistic_salary")

        # ── 9. Profile completeness vs data richness ─────────────
        completeness = row.get("profile_completeness", 0)
        # High completeness score but very few skills
        if completeness > 90 and skill_count < 3:
            flags.append("completeness_skill_mismatch")

        # ── 10. Response rate anomaly ────────────────────────────
        rr = row.get("recruiter_response_rate", 0)
        if rr > 1.0:
            flags.append("response_rate_exceeds_100pct")

        # ── Compute trust score ──────────────────────────────────
        # Each flag reduces trust slightly
        # Severe flags reduce more than minor ones
        severe_flags = {
            "impossible_experience_35+",
            "future_last_active_date",
            "active_before_signup",
            "perfect_assessment_scores",
            "endorsement_connection_mismatch",
            "response_rate_exceeds_100pct",
        }

        trust_reduction = 0.0
        for flag in flags:
            if any(flag.startswith(sf) for sf in severe_flags):
                trust_reduction += 0.15  # severe: -15% trust
            else:
                trust_reduction += 0.07  # minor: -7% trust

        trust_score = max(1.0 - trust_reduction, 0.3)  # floor at 30%

        # Write back
        df.at[idx, "anomaly_flags"] = flags
        df.at[idx, "anomaly_count"] = len(flags)
        df.at[idx, "trust_score"] = round(trust_score, 4)

    # Summary
    total_anomalies = (df["anomaly_count"] > 0).sum()
    print(f"Anomaly detection complete.")
    print(f"Candidates with at least 1 flag: {total_anomalies:,} "
          f"({total_anomalies/len(df)*100:.1f}%)")
    print(f"Average trust score: {df['trust_score'].mean():.3f}")

    return df
