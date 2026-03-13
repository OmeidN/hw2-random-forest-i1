# Step-by-Step Instructions (Beginner-Friendly)

This guide explains exactly what to do after the code is generated, including if you have never used Jupyter before.

---

## 1. Open the Project

- **In Cursor or VS Code:** File → Open Folder → select the project folder (`hw2_random_forest_i1` or `hw2-random-forest-i1`).
- The left sidebar will show the folder structure (data, notebooks, src, etc.).

---

## 2. Create a Virtual Environment (Recommended)

A virtual environment keeps this project’s packages separate from other Python projects.

**Windows (PowerShell or Command Prompt):**

```powershell
cd path\to\hw2-random-forest-i1
python -m venv venv
```

**Activate on Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Activate on Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**Mac/Linux:**

```bash
cd path/to/hw2-random-forest-i1
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line when it’s active.

---

## 3. Install Requirements

With the virtual environment **activated**, from the **project root** folder run:

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, and openpyxl.

---

## 4. Place the CSV in the Right Folder

1. Locate your file: **Original training DB i1 cluster.csv**
2. Put it inside: **data/raw/**
   - Full path should look like: `data/raw/Original training DB i1 cluster.csv`
3. If the `raw` folder doesn’t exist, create it first.

---

## 5. Open the Notebook

- In Cursor/VS Code: click **notebooks/hw2_rf_pipeline.ipynb** in the file explorer.
- The notebook opens as a series of **cells** (boxes). There are two main types:
  - **Markdown cells:** Plain text and headings (not run as code).
  - **Code cells:** Python code that you can run.

---

## 6. What a Code Cell and Markdown Cell Are

- **Code cell:** Contains Python (e.g. `import pandas`, `load_data(...)`). When you run it, the code executes and output appears below.
- **Markdown cell:** Contains formatted text (headings, lists). It’s for reading; you can run it to “render” the text, but it doesn’t run Python.

---

## 7. How to Run a Cell

- Click inside a cell so it’s selected.
- **Run one cell:** Press **Shift+Enter** (runs the cell and moves to the next), or use the ▶ Run button above the cell.
- **Run all cells from the top:** Use the menu **Run → Run All** (or equivalent in your editor). This runs every cell in order, which is what you want for the full workflow.

---

## 8. What Order to Run the Notebook In

**Always run from top to bottom.**

1. Run the first cell (imports and path setup).
2. Then run each following cell in order. The notebook is designed so that later cells use variables from earlier ones (e.g. `df_original`, `best_model`).
3. Easiest: use **Run All** once so the whole pipeline runs in the correct order.

---

## 9. What Output Files Should Appear If Everything Works

After running the whole notebook you should see:

- **data/processed/training_db.csv** — training set
- **data/processed/verification_db.csv** — 2 held-out samples
- **data/outputs/oob_results.csv** — OOB grid results
- **data/outputs/cv_fold_results.csv** — 3-fold CV per-fold results
- **data/outputs/top10_features.csv** — top 10 features by Gini importance
- **data/outputs/verification_predictions.csv** — predictions on the 2 verification samples
- **reports/figures/top10_features.png** — bar chart of top 10 features

If any of these are missing, check that you ran all cells and that the CSV was in **data/raw/** with the correct name.

---

## 10. If You Get Import Errors

**“No module named 'src'” or “No module named 'pandas'”:**

- Make sure the **virtual environment is activated** (you see `(venv)` in the terminal).
- Install requirements again: `pip install -r requirements.txt`
- In the notebook, run the project from the **project root** (the folder that contains both `src` and `notebooks`). The first cell adds the project root to `sys.path` so `from src.config import ...` works. If you open the notebook from a different folder, set your working directory to the project root or run the notebook from the root.

**“FileNotFoundError” for the CSV:**

- Confirm the file is at **data/raw/Original training DB i1 cluster.csv** (name and path must match).

---

## 11. If the Label Column Is "Label" Instead of "label"

The code is written to accept both. The notebook uses `detect_label_column(df_original)`, which looks for a column named `label` or `Label` (case-insensitive). You don’t need to change anything; it will use whichever column exists. If your CSV uses a different name (e.g. `Class`), you would need to rename that column to `label` or `Label` in the CSV, or adjust the code to look for that name.

---

## 12. After the Notebook Runs

- Check **data/outputs/** and **reports/figures/** for the CSVs and the top-10 plot.
- Use the notebook output (tables, metrics, feature list, verification predictions) to write your homework report.

---

## 13. Using the Notebook Output for Your Homework Report PDF

- You can export the notebook to HTML or PDF (e.g. Jupyter: File → Download as → PDF, or use “Print” to PDF from the browser).
- Copy tables and numbers from the notebook into your report. The notebook shows accuracy and other metrics to 5 decimal places as required.
- **Important:** The report still needs **your own explanatory text**. Don’t hand in only raw output: describe what you did (audit, split, OOB vs CV, how you chose the best model, what the top features and ground-truth overlap mean, and what the verification predictions show).

---

## 14. Quick Checklist

- [ ] Project opened in Cursor or VS Code  
- [ ] Virtual environment created and activated  
- [ ] `pip install -r requirements.txt` run  
- [ ] **Original training DB i1 cluster.csv** placed in **data/raw/**  
- [ ] Notebook **hw2_rf_pipeline.ipynb** opened  
- [ ] All cells run in order (or Run All)  
- [ ] Output CSVs and figure appear in **data/outputs/** and **reports/figures/**  
- [ ] Report written with your own explanations, not just raw output  

If you hit a step that doesn’t work, re-read the section for that step and the “If you get import errors” and “If the label column…” sections above.
