import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_candidates(path: str) -> pd.DataFrame:
    """Load all candidates from jsonl file into a flat DataFrame."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading candidates"):
            raw = json.loads(line)

            p = raw.get("profile", {})
            signals = raw.get("redrob_signals", {})
            skills = raw.get("skills", [])
            career = raw.get("career_history", [])
            education = raw.get("education", [])

            # Skills
            skill_names = [
                s.get("name", "").lower()
                for s in skills
                if s.get("name")
            ]

            skill_advanced = [
                s.get("name", "").lower()
                for s in skills
                if s.get("proficiency") == "advanced"
            ]

            total_endorsements = sum(
                s.get("endorsements", 0)
                for s in skills
            )

            # Career
            all_descriptions = " ".join(
                c.get("description", "")
                for c in career
            )

            all_titles = " ".join(
                c.get("title", "")
                for c in career
            )

            industries = list(
                set(
                    c.get("industry", "")
                    for c in career
                    if c.get("industry")
                )
            )

            career_jobs_count = len(career)

            # Consulting-only flag
            consulting_firms = {
                "tcs",
                "infosys",
                "wipro",
                "accenture",
                "cognizant",
                "capgemini",
                "hcl",
                "mindtree",
                "tech mahindra",
            }

            companies_worked = {
                c.get("company", "").lower()
                for c in career
                if c.get("company")
            }

            only_consulting = (
                len(companies_worked) > 0
                and companies_worked.issubset(consulting_firms)
            )

            # Education
            tier_map = {
                "tier_1": 1,
                "tier_2": 2,
                "tier_3": 3,
                "tier_4": 4,
            }

            tiers = [
                tier_map.get(
                    e.get("tier", "tier_4"),
                    4,
                )
                for e in education
            ]

            best_tier = min(tiers) if tiers else 4

            # Salary
            sal = signals.get(
                "expected_salary_range_inr_lpa",
                {},
            )

            sal_min = sal.get("min", 0) or 0
            sal_max = sal.get("max", 0) or 0

            if sal_min > sal_max:
                sal_min, sal_max = sal_max, sal_min

            # Full text
            full_text = " ".join(
                [
                    p.get("headline", ""),
                    p.get("summary", ""),
                    all_titles,
                    all_descriptions,
                    " ".join(skill_names),
                ]
            )

            # Parse date once
            last_active_date = signals.get(
                "last_active_date",
                None,
            )

            last_active_dt = pd.to_datetime(
                last_active_date,
                errors="coerce",
            )

            records.append(
                {
                    # Identity
                    "candidate_id": raw["candidate_id"],
                    "name": p.get(
                        "anonymized_name",
                        "",
                    ),

                    # Profile
                    "headline": p.get("headline", ""),
                    "summary": p.get("summary", ""),
                    "location": p.get("location", ""),
                    "country": p.get("country", ""),
                    "years_of_experience": p.get(
                        "years_of_experience",
                        0,
                    ),
                    "current_title": p.get(
                        "current_title",
                        "",
                    ),
                    "current_company": p.get(
                        "current_company",
                        "",
                    ),
                    "current_industry": p.get(
                        "current_industry",
                        "",
                    ),

                    # Skills
                    "skill_names": skill_names,
                    "skill_advanced": skill_advanced,
                    "skill_count": len(skill_names),
                    "advanced_skill_count": len(
                        skill_advanced
                    ),
                    "total_endorsements": total_endorsements,
                    # Career
                    "all_descriptions": all_descriptions,
                    "all_titles": all_titles,
                    "industries": industries,
                    "career_jobs_count": career_jobs_count,
                    "only_consulting": only_consulting,
                    # Preserve raw data
                    "career_history": career,
                    "education_history": education,
                    # Education
                    "best_edu_tier": best_tier,
                    # Text
                    "full_text": full_text,
                    # Dates
                    "last_active_date": last_active_date,
                    "last_active_dt": last_active_dt,
                    # Signals
                    "profile_completeness": signals.get(
                        "profile_completeness_score",0,),
                    "open_to_work": signals.get(
                        "open_to_work_flag", False,),
                    "recruiter_response_rate": signals.get(
                        "recruiter_response_rate",0,),
                    "avg_response_time_hours": signals.get("avg_response_time_hours",999,),
                    "github_activity_score": signals.get("github_activity_score",-1,),
                    "interview_completion_rate": signals.get("interview_completion_rate",0,),
                    "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1,),
                    "notice_period_days": signals.get("notice_period_days", 90,),
                    "willing_to_relocate": signals.get("willing_to_relocate",False,),
                    "preferred_work_mode": signals.get( "preferred_work_mode","",),
                    "profile_views_30d": signals.get("profile_views_received_30d", 0,),
                    "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d",0, ),
                    "applications_30d": signals.get("applications_submitted_30d",0, ),
                    "verified_email": signals.get("verified_email", False, ),
                    "verified_phone": signals.get("verified_phone",  False, ),
                    "linkedin_connected": signals.get("linkedin_connected", False, ),
                    "connection_count": signals.get("connection_count", 0, ),
                    "sal_min": sal_min,
                    "sal_max": sal_max,
                    "skill_assessment_scores": signals.get("skill_assessment_scores",{},),
                    # Preserve all signals
                    "raw_signals": signals,
                }
            )
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} candidates successfully.")
    return df


def load_jd(path: str) -> str:
    """Load job description as plain text."""
    path = Path(path)
    if path.suffix == ".docx":
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            print("python-docx not installed, reading as txt fallback")
            return path.read_text(encoding="utf-8", errors="ignore")
    return path.read_text(encoding="utf-8", errors="ignore")


if __name__ == "__main__":
    df = load_candidates("data/candidates.jsonl")
    print(df.head(3))
    print(df.columns.tolist())
