# Deploying the Gradio app to Hugging Face Spaces

## 1. Train and save a model (in Colab)
After training, save the best model to a checkpoint:

```python
import eda, train
wide = eda.group_rare_species(eda.to_wide(eda.load_long()))

# train a deployable model on (almost) all data
m, info = train.train_final_model(
    wide, kind="resnet18",
    model_kwargs={"pretrained": True, "freeze_backbone": False},
    train_kwargs={"max_epochs": 30, "patience": 8, "lr": 1e-4},
)
train.save_checkpoint(m, info, "models/best_model.pt")
```

Download `models/best_model.pt` (and `best_model.json`) from Colab.

## 2. Create a Hugging Face Space
- New Space -> SDK: Gradio.
- Upload: `app.py` (rename from src/app.py to repo root), the `src/` modules it
  imports (config.py, dataset.py, model.py, train.py, splits.py), the
  `models/best_model.pt`, and `requirements.txt` (use requirements_app.txt).
- Ensure `CHECKPOINT_PATH` in app.py points to `models/best_model.pt`.

## 3. Run locally first
```
pip install -r requirements_app.txt
python src/app.py
```
Opens a local Gradio URL. Upload a quadrat photo to test.

## Notes
- Image-only model: this is the end-user (farmer) scenario. Metadata-based
  models are stronger but require NDVI/height, unavailable in the field.
- The app loads the model lazily on first prediction.
