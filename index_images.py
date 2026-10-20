import re, random, json
from pathlib import Path
from fastembed import ImageEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

DATA_DIR = Path(r"archive/PlantVillage")  # adjust to your path
HOLDOUT = 0.15
MAX_PER_CLASS = 300
random.seed(42)

model = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")   # downloads ~350 MB once, then cached
client = QdrantClient(path="qdrant_db")

COLLECTION = "plant_diseases_clip"
if not client.collection_exists(COLLECTION):
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

def parse_label(folder_name):
    m = re.match(r"^(Pepper_+bell|[A-Za-z]+)_+(.+)$", folder_name)
    if not m:
        return None
    plant = re.sub(r"_+", " ", m.group(1))     # fixes "Pepper  bell" double space
    disease = re.sub(r"_+", " ", m.group(2))
    return plant, disease

solutions = {}
if Path("solutions.json").exists():
    solutions = json.loads(Path("solutions.json").read_text(encoding="utf-8"))

test_set, pid = [], 0
for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    parsed = parse_label(class_dir.name)
    if parsed is None:
        print("Skipping folder (not a class):", class_dir.name)
        continue
    plant, disease = parsed
    images = sorted(class_dir.glob("*.[jJ][pP][gG]"))
    random.shuffle(images)
    if MAX_PER_CLASS:
        images = images[:MAX_PER_CLASS]

    n_test = int(len(images) * HOLDOUT)
    test_set += [(str(p), plant, disease) for p in images[:n_test]]
    train = images[n_test:]
    info = solutions.get(f"{plant}|{disease}", {})

    vecs = list(model.embed([str(p) for p in train], batch_size=16))
    points = []
    for p, v in zip(train, vecs):
        points.append(PointStruct(
            id=pid, vector=v.tolist(),
            payload={"plant": plant, "disease": disease, "path": str(p),
                     "symptoms": info.get("symptoms", ""),
                     "solution": info.get("solution", "")},
        ))
        pid += 1
    client.upload_points(COLLECTION, points)   # upload per class so memory stays low
    print(f"{plant} / {disease}: {len(train)} indexed, {n_test} held out")

json.dump(test_set, open("test_set.json", "w"))
print("Done:", pid, "points indexed")