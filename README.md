# 🌿 Plant Doctor

Upload a photo of a sick plant and get a likely diagnosis plus suggestions for how to save it.

Plant Doctor identifies the plant, then searches a vector database of labelled plant-disease images to find the closest matching diseases. Everything runs locally with no PyTorch dependency, so it stays light to install and deploy.

**Live app:** `<add your deployed URL here>`

---

## How it works

1. **Upload**: the user uploads a leaf/plant photo in the Streamlit sidebar and can optionally type a question.
2. **Embed**: the image is converted to a CLIP embedding using [fastembed](https://github.com/qdrant/fastembed) (ONNX, no PyTorch).
3. **Identify the plant**: CLIP zero-shot matching estimates the plant species (tomato, potato, bell pepper).
4. **Retrieve**: Qdrant (local mode) finds the nearest labelled disease images, filtered to the identified plant.
5. **Answer**: the app shows the top matching diseases with similarity scores and the recommended solution.

```
Photo ──► CLIP embedding (fastembed) ──► zero-shot plant ID
                     │                          │
                     └──► Qdrant similarity search (filtered by plant)
                                    │
                                    ▼
                    Top disease matches + solutions (Streamlit UI)
```

## Tech stack

| Component | Choice |
|---|---|
| UI | Streamlit |
| Embeddings | CLIP via fastembed (ONNX) |
| Vector DB | Qdrant, local mode (`qdrant_db/`) |
| Image handling | Pillow |
| Dataset | PlantVillage (15 classes: pepper bell, potato, tomato) |

## Project structure

```
Plant dieases diagnosis app/
├── app.py                    # Streamlit app (the deployed entry point)
├── index_images.py           # Builds the Qdrant index from PlantVillage
├── evaluate_real_world.py    # Evaluates on your own photos in real_world_test/<plant>/
├── diagnose_plant_id.py      # Compares zero-shot vs trained plant identification
├── score_check.py            # Compares similarity scores: dataset vs real-world photos
├── train_plant_classifier.py # (Experimental) logistic-regression plant classifier
├── test_set.json             # Held-out images created at indexing time
├── qdrant_db/                # Local vector database
├── real_world_test/          # Your own test photos, one folder per plant
└── archive/PlantVillage/     # Dataset (not committed)
```

## Setup

```bash
pip install streamlit fastembed qdrant-client pillow
```

Download PlantVillage from Kaggle and unzip it to `archive/PlantVillage/`, so each class is a folder (e.g. `Tomato_Early_blight`).

## Usage

**1. Build the index** (once):

```bash
python index_images.py
```

This indexes up to 300 images per class into `qdrant_db/` and saves the held-out images to `test_set.json`.

**2. Run the app:**

```bash
streamlit run app.py
```

Open the sidebar, upload a plant photo, optionally add a question, and read the result.

## Deployment notes

- Commit `qdrant_db/` (or rebuild the index on startup) so the deployed app has data to search. Local-mode Qdrant is file-based, so the host needs a persistent or bundled copy.
- Do not commit the raw PlantVillage dataset; only the built index is needed at runtime.
- Pin versions in `requirements.txt`:
  ```
  streamlit
  fastembed
  qdrant-client
  pillow
  ```
- Newer Streamlit versions deprecate `use_container_width`; use `width="stretch"` instead if you see that warning.

## Evaluation results

| Run | Indexed / class | Held-out images | Top-1 | Top-3 |
|---|---|---|---|---|
| 1 | 85 | 225 | 80.4% | 93.3% |
| 2 | up to 300 | 652 | 85.3% | 95.6% |

Weakest classes: Tomato late blight, early blight, Septoria leaf spot and Target Spot.

## Known limitations

- **Lab vs real-world gap.** PlantVillage images are clean, single-leaf photos on plain backgrounds. Best-match similarity is a median of 0.967 on held-out dataset images but only about 0.81 on real-world photos, so disease matches on real photos are less reliable (neighbours are often spread across several diseases).
- **Plant identification.** Zero-shot CLIP got 5/6 real-world photos right, versus 2/6 for a trained classifier that scored 99.5% on PlantVillage but overfit to its look. Zero-shot is used in the app for that reason.
- **Limited coverage.** Only the 15 PlantVillage classes (pepper bell, potato, tomato) are supported.
- This is a decision aid, not a substitute for an agronomist. Treat low similarity scores with caution.

## Roadmap

- Add real-world, in-the-wild images to the index to close the dataset gap
- Show a low-confidence warning when similarity falls below a threshold
- Support more plants and diseases
- Add re-ranking or fine-tuning for the hard tomato classes

## Acknowledgements

- [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- [fastembed](https://github.com/qdrant/fastembed) and [Qdrant](https://qdrant.tech/)
- [Streamlit](https://streamlit.io/)