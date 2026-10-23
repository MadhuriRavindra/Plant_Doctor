# 🌿 Plant Doctor (v1)

Upload a photo of a sick plant and get a likely diagnosis plus a suggested solution.

Plant Doctor embeds your photo with CLIP and searches a vector database of labelled plant-disease images to find the closest matching diseases. It runs without PyTorch, so it stays light to install and deploy.

**Live app:** `<add your deployed URL here>`

---

## How it works

1. **Upload** a leaf/plant photo in the Streamlit app.
2. **Choose the plant** from the "Which plant is it?" selector (default: "Not sure") and optionally type a question.
3. **Embed** the image with CLIP via [fastembed](https://github.com/qdrant/fastembed) (ONNX, no PyTorch).
4. **Retrieve** the nearest labelled disease images from Qdrant (local mode).
5. **Answer**: the app shows the top matching diseases and the recommended solution.

```
Photo ──► CLIP embedding (fastembed) ──► Qdrant similarity search
                                              │
                                              ▼
                              Top disease matches + solution (Streamlit)
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
├── app.py            # Streamlit app (entry point)
├── index_images.py   # Builds the Qdrant index from PlantVillage
├── test_set.json     # Held-out images created at indexing time
├── qdrant_db/        # Local vector database
└── archive/PlantVillage/   # Dataset (not needed at runtime)
```

## Setup

```bash
pip install streamlit fastembed qdrant-client pillow
```

Download PlantVillage from Kaggle and unzip it to `archive/PlantVillage/`, so each class is its own folder.

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

## Deployment notes

- Bundle `qdrant_db/` with the app (or rebuild the index on startup). Local-mode Qdrant is file-based, so the host needs a copy of it.
- Don't ship the raw PlantVillage dataset; only the built index is needed at runtime.
- `requirements.txt`:
  ```
  streamlit
  fastembed
  qdrant-client
  pillow
  ```

## Evaluation results

Retrieval accuracy on held-out PlantVillage images:

| Run | Indexed / class | Held-out images | Top-1 | Top-3 |
|---|---|---|---|---|
| 1 | 85 | 225 | 80.4% | 93.3% |
| 2 | up to 300 | 652 | 85.3% | 95.6% |

Weakest classes: Tomato late blight, early blight, Septoria leaf spot and Target Spot.

## Known limitations

- Covers only the 15 PlantVillage classes (pepper bell, potato, tomato).
- PlantVillage photos are clean, single-leaf images on plain backgrounds, so photos taken in the field may match less reliably.
- Some tomato diseases look similar and are confused more often.
- This is a decision aid, not a substitute for an agronomist.

## Acknowledgements

- [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- [fastembed](https://github.com/qdrant/fastembed) and [Qdrant](https://qdrant.tech/)
- [Streamlit](https://streamlit.io/)