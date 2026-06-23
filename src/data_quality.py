import pandas as pd
import numpy as np
from datetime import datetime


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast vectorized anomaly detection.

    Adds:
        anomaly_count
        trust_score
    """

    print("Running data quality checks...")

    df = df.copy()

    # Experience anomalies
    impossible_experience = df["years_of_experience"] > 35

    entry_title_high_exp = (
        df["current_title"]
        .fillna("")
        .str.lower()
        .str.contains(
            r"intern|trainee|fresher|junior|associate",
            regex=True
        )
        & (df["years_of_experience"] > 8)
    )

    # Skill anomalies
    skill_inflation_count = df["advanced_skill_count"] > 15

    skill_inflation_ratio = (
        (df["advanced_skill_count"] / df["skill_count"].clip(lower=1) > 0.8)
        & (df["skill_count"] > 5)
    )

    # Endorsement anomalies
    endorsement_connection_mismatch = (
        (df["connection_count"] < 20)
        & (df["total_endorsements"] > 200)
    )

    # Date anomalies
    future_last_active = df["last_active_dt"] > pd.Timestamp.now()

    # Salary anomalies
    missing_salary_experienced = (
        (df["years_of_experience"] > 5)
        & (df["sal_min"] == 0)
        & (df["sal_max"] == 0)
    )

    unrealistic_salary = df["sal_max"] > 500

    # Completeness anomalies
    completeness_skill_mismatch = (
        (df["profile_completeness"] > 90)
        & (df["skill_count"] < 3)
    )

    # Response rate anomalies
    response_rate_invalid = df["recruiter_response_rate"] > 1.0

    # Build anomaly matrix
    anomaly_matrix = pd.DataFrame({
        "impossible_experience": impossible_experience,
        "entry_title_high_exp": entry_title_high_exp,
        "skill_inflation_count": skill_inflation_count,
        "skill_inflation_ratio": skill_inflation_ratio,
        "endorsement_connection_mismatch": endorsement_connection_mismatch,
        "future_last_active": future_last_active,
        "missing_salary_experienced": missing_salary_experienced,
        "unrealistic_salary": unrealistic_salary,
        "completeness_skill_mismatch": completeness_skill_mismatch,
        "response_rate_invalid": response_rate_invalid
    })

    # Count anomalies
    df["anomaly_count"] = anomaly_matrix.sum(axis=1)

    # Severe anomalies
    severe_count = (
        impossible_experience.astype(int)
        + endorsement_connection_mismatch.astype(int)
        + future_last_active.astype(int)
        + response_rate_invalid.astype(int)
    )

    # Minor anomalies
    minor_count = df["anomaly_count"] - severe_count

    # Trust score
    df["trust_score"] = (
        (0.85 ** severe_count)
        * (0.95 ** minor_count)
    ).clip(lower=0.30).round(4)

    # Optional: store anomaly flags
    anomaly_cols = anomaly_matrix.columns.tolist()

    df["anomaly_flags"] = anomaly_matrix.apply(
        lambda row: [col for col in anomaly_cols if row[col]],
        axis=1
    )

    print("Anomaly detection complete.")
    print(
        f"Candidates with anomalies: "
        f"{(df['anomaly_count'] > 0).sum():,} "
        f"({((df['anomaly_count'] > 0).mean() * 100):.1f}%)"
    )
    print(f"Average trust score: {df['trust_score'].mean():.3f}")

    return df