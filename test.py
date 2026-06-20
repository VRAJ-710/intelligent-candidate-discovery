from src.data_loader import load_candidates
from src.scorer import score_all
from src.reasoning import generate_reasoning

df = load_candidates('data/candidates.jsonl')
result = score_all(df)

top5 = result.head(5)
for _, row in top5.iterrows():
    print(row['candidate_id'], '|', row['score'])
    print(generate_reasoning(row))
    print()
