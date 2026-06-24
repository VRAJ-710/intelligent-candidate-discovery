from datetime import datetime


def generate_reasoning(row) -> str:
    """
    Generate a human-readable 1-2 sentence reasoning for why this
    candidate is ranked where they are. Uses only real data from the row.
    """
    parts = []
    title = row.get("current_title", "Professional")
    yoe = row.get("years_of_experience", 0)
    company = row.get("current_company", "")
    parts.append(f"{title} with {yoe:.1f} years of experience at {company}.")

    from src.scorer import CORE_AI_SKILLS
    skill_names = set(row.get("skill_names", []))
    matched = skill_names & CORE_AI_SKILLS
    advanced = set(row.get("skill_advanced", [])) & CORE_AI_SKILLS

    if advanced:
        top = list(advanced)[:3]
        parts.append(f"Advanced proficiency in {', '.join(top)}.")
    elif matched:
        top = list(matched)[:3]
        parts.append(f"Relevant skills include {', '.join(top)}.")
    else:
        parts.append("Limited AI/ML skill match for this JD.")

    signals = []

    try:
        last_active = datetime.strptime(row["last_active_date"], "%Y-%m-%d")
        days_ago = (datetime.now() - last_active).days
        if days_ago <= 30:
            signals.append("active in last 30 days")
        elif days_ago <= 90:
            signals.append("active in last 90 days")
        else:
            signals.append(f"last active {days_ago} days ago")
    except Exception:
        pass

    rr = row.get("recruiter_response_rate", 0)
    if rr >= 0.6:
        signals.append(f"high recruiter response rate ({rr:.0%})")
    elif rr <= 0.2:
        signals.append(f"low recruiter response rate ({rr:.0%})")

    gh = row.get("github_activity_score", -1)
    if gh > 20:
        signals.append(f"GitHub activity score {gh:.0f}")

    np_days = row.get("notice_period_days", 90)
    if np_days <= 30:
        signals.append(f"notice period {np_days} days")
    elif np_days >= 120:
        signals.append(f"long notice period ({np_days} days)")

    otw = row.get("open_to_work", False)
    if otw:
        signals.append("open to work")

    if row.get("only_consulting", False):
        signals.append("only consulting background — JD flag")

    assessments = row.get("skill_assessment_scores", {})
    if assessments:
        avg = sum(assessments.values()) / len(assessments)
        signals.append(f"assessment avg {avg:.0f}/100")

    if signals:
        parts.append("Signals: " + "; ".join(signals) + ".")

    location = row.get("location", "")
    country = row.get("country", "")
    relocate = row.get("willing_to_relocate", False)
    if location:
        loc_note = f"Based in {location}, {country}"
        if relocate:
            loc_note += " (willing to relocate)"
        parts.append(loc_note + ".")

    full = " ".join(parts)

    if len(full) > 300:
        full = full[:297] + "..."

    return full
