# E-Shop Recommendation Dashboard

This project contains:

- A Streamlit dashboard.
- A Popular + Discovery recommendation model.
- Offline Top-10 model evaluation.

## 1. Required Data Files

Place these files inside the `dashboard/data` folder:

- `dashboard/data/reviews_clean_no_exact_duplicates.csv`
- `dashboard/data/products_clean.csv`
- `dashboard/data/user_summary.csv`

## 2. Download the Project

Clone the repository and enter its folder:

```bash
git clone https://github.com/Kw2343/dashboard_stuido6.git
cd dashboard
```

If you downloaded a ZIP file instead, extract it and open a terminal in the
extracted project folder.

## 3. Install on macOS

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r dashboard/requirements_dashboard.txt
```

## 4. Install on Windows

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it in Command Prompt:

```bat
.venv\Scripts\activate
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r dashboard\requirements_dashboard.txt
```

## 5. Run the Dashboard

Run this command from the project root:

```bash
python -m streamlit run dashboard/app.py
```

Open the URL shown in the terminal. The default address is:

[http://localhost:8501](http://localhost:8501)

Stop the dashboard with `Ctrl+C`.

## 6. Evaluate the Popularity Model

The evaluator uses:

- A chronological 80/20 train/test split for users with at least five
  interactions.
- The actual scoring functions from dashboard/tabs/popularity.py.
- The same global Top-10 list for every user.
- Eight Popular products and two Discovery products by default.
- Every held-out interaction as a relevant purchase, regardless of rating.

Run on macOS or Windows:

```bash
python dashboard/model-evaluation.py --model popularity
```

The evaluator reports:

- Recall@10
- Precision@10
- Hit Rate@10
- MRR@10
- MAP@10
- NDCG@10
- Embedding-based Diversity@10
- Popularity Bias@10
- Coverage@10

Results are saved to:

- `dashboard/evaluation_results/popularity_metrics.csv`
- `dashboard/evaluation_results/popularity_user_details.csv`

The train/test split and product embeddings are cached in:

`dashboard/evaluation_results/cache/`

The first evaluation may take longer because the sentence embedding model must
be downloaded and loaded. Later evaluations reuse the cached data and
embeddings.

For simple explanations and examples of every metric, read:

[`dashboard/MODEL_EVALUATION_METRICS.md`](dashboard/MODEL_EVALUATION_METRICS.md)

## 7. Evaluation Options

Change the recommendation list size:

```bash
python dashboard/model-evaluation.py --model popularity --top-k 20
```

Change the Discovery share to 30 percent:

```bash
python dashboard/model-evaluation.py --model popularity --discovery-share 0.30
```

Change the Discovery quality gate:

```bash
python dashboard/model-evaluation.py --model popularity --min-discovery-rating 4.0 --min-discovery-reviews 10
```

Evaluate recommendations produced by another model:

```bash
python dashboard/model-evaluation.py --recommendations-csv path/to/recommendations.csv
```

The recommendations CSV must contain:

```csv
user_id,parent_asin
```

It may also contain `rank` or `predicted_score`.

## 8. Deactivate the Environment

When finished, run:

```bash
deactivate
```
