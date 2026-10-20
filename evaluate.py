import json
from collections import defaultdict, Counter
from fastembed import ImageEmbedding
from qdrant_client import QdrantClient

COLLECTION = "plant_diseases_clip"
K = 5

model = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
client = QdrantClient(path="qdrant_db")

test_set = json.load(open("test_set.json"))   # [path, plant, disease]
vecs = list(model.embed([t[0] for t in test_set], batch_size=16))

top1 = top3 = 0
per_class = defaultdict(lambda: [0, 0])       # label -> [correct, total]
confusions = Counter()

for (path, plant, disease), v in zip(test_set, vecs):
    hits = client.query_points(COLLECTION, query=v.tolist(), limit=K).points

    # aggregate by (plant, disease): sum of similarity scores across the top-K hits
    scores = defaultdict(float)
    for h in hits:
        scores[(h.payload["plant"], h.payload["disease"])] += h.score
    ranked = [lab for lab, _ in sorted(scores.items(), key=lambda x: -x[1])]

    truth = (plant, disease)
    label = f"{plant} / {disease}"
    per_class[label][1] += 1
    if ranked[0] == truth:
        top1 += 1
        per_class[label][0] += 1
    else:
        confusions[(label, f"{ranked[0][0]} / {ranked[0][1]}")] += 1
    if truth in ranked[:3]:
        top3 += 1

n = len(test_set)
print(f"\nTest images: {n}")
print(f"Top-1 accuracy: {top1 / n:.1%}")
print(f"Top-3 accuracy: {top3 / n:.1%}\n")

print("Per-class top-1:")
for label, (c, t) in sorted(per_class.items(), key=lambda x: x[1][0] / x[1][1]):
    print(f"  {c / t:5.0%}  {label}  ({c}/{t})")

print("\nMost common mistakes (true -> predicted):")
for (true, pred), cnt in confusions.most_common(8):
    print(f"  {cnt}x  {true}  ->  {pred}")