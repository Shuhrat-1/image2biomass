# Deploying this Space

## Files in this folder (hf_space/)
```
app.py              Gradio entry point (HF runs this)
README.md           Space config (YAML header) + description
requirements.txt    Dependencies
src/                Model modules (config, dataset, train, baseline, model, splits)
models/             <- you add best_model.pt here (see below)
```

## Steps
1. **Get the checkpoint.** After training in Colab, download `best_model.pt`
   (from the Colab `outputs/` folder) and place it here as
   `hf_space/models/best_model.pt`.

2. **Create a Space.** On huggingface.co: New → Space → SDK: Gradio → name it
   (e.g. `pasture-biomass`). Choose CPU basic (free) — the model is small.

3. **Upload files.** Either:
   - Drag-and-drop all of `hf_space/` contents into the Space's Files tab
     (app.py, README.md, requirements.txt, src/, models/best_model.pt), or
   - Use git:
     ```
     git clone https://huggingface.co/spaces/<user>/pasture-biomass
     cp -r hf_space/* pasture-biomass/
     cd pasture-biomass
     git add . && git commit -m "Add biomass app" && git push
     ```

4. **Wait for build.** The Space installs requirements and launches app.py.
   First prediction lazy-loads the model. Done — you get a public URL.

## Notes
- The checkpoint must be < the Space file limit; best_model.pt (ResNet18) is
  ~45 MB, well within limits.
- If the build fails, check the Space logs tab (usually a requirements version).
- The Space needs no data — only the trained checkpoint.
