import pandas as pd
from src.data_loader import load_candidates
from src.scorer import score_all
from src.reasoning import generate_reasoning
from tqdm import tqdm
import os


def run_pipeline(
    candidates_path: str = "data/candidates.jsonl",
    output_path: str = "data/output/submission.csv",
    top_k: int = 100,
    team_id: str = "team_001"
):
    print("=" * 55)
    print("  Intelligent Candidate Discovery Pipeline")
    print("=" * 55)

    # Step 1 — Load
    print("\n[1/4] Loading candidates...")
    df = load_candidates(candidates_path)

    # Step 2 — Score
    print("\n[2/4] Scoring candidates...")
    scored = score_all(df)

    # Step 3 — Generate reasoning for top_k
    print(f"\n[3/4] Generating reasoning for top {top_k} candidates...")
    top_k = min(top_k, len(scored))
    top = scored.head(top_k).copy()

    reasonings = []
    for _, row in tqdm(top.iterrows(), total=len(top)):
        reasonings.append(generate_reasoning(row))
    top["reasoning"] = reasonings

    # Step 4 — Build submission CSV
    print("\n[4/4] Building submission CSV...")
    top["rank"] = range(1, top_k + 1)

    scores = top["score"].values.copy()
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            scores[i] = scores[i - 1]
    top["score"] = [round(float(s), 4) for s in scores]

    top = top.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    top["rank"] = range(1, top_k + 1)

    submission = top[["candidate_id", "rank", "score", "reasoning"]].copy()

    assert len(submission) == top_k
    assert list(submission["rank"]) == list(range(1, top_k + 1))
    assert submission["candidate_id"].nunique() == top_k

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = output_path.replace("submission.csv", f"{team_id}.csv")
    submission.to_csv(output_file, index=False, encoding="utf-8")

    print("\n" + "=" * 55)
    print(f"  Submission saved to: {output_file}")
    print(f"  Total candidates scored: {len(scored):,}")
    print(f"  Top 100 selected.")
    print(f"  Score range: {submission['score'].iloc[0]} → {submission['score'].iloc[-1]}")
    print("=" * 55)

    print("\nTop 10 preview:")
    print(submission[["candidate_id", "rank", "score"]].head(10).to_string(index=False))

    return submission


if __name__ == "__main__":
    run_pipeline()
