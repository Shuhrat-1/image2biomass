# One-time GitHub repo setup

Run these in the project root (`image2biomass/`).

## 1. Organize notebooks into a folder (optional but cleaner)
```cmd
mkdir notebooks
move eda_first.ipynb notebooks\
move baseline_tabular.ipynb notebooks\
move train_colab.ipynb notebooks\
```
Note: if you move notebooks into `notebooks/`, they still work — the path cell
already handles being run from a subfolder (`ROOT.parent / "src"`).

## 2. Initialize Git and commit
```cmd
git init
git add .gitignore README.md project_proposal.md requirements.txt requirements_app.txt DEPLOY.md
git add src/ notebooks/ data/raw/labelled.csv
git commit -m "Initial commit: EDA, baselines, CNN pipeline, Gradio app"
```

The `.gitignore` excludes the ~1 GB of images and model checkpoints, so only
code, notebooks, and the small CSV are committed.

## 3. Create the repo on GitHub and push
- Create an empty repo on github.com (no README, since you have one).
```cmd
git remote add origin https://github.com/<your-username>/image2biomass.git
git branch -M main
git push -u origin main
```

## 4. Submit in Moodle
Submit the link to your proposal file, e.g.:
`https://github.com/<your-username>/image2biomass/blob/main/project_proposal.md`

## Verify before pushing
```cmd
git status            # confirm images are NOT staged
git ls-files | findstr ".jpg"   # should return nothing
```
If any .jpg appears, the .gitignore isn't catching them — check paths.
