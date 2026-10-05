# California House Price Predictor

An end-to-end, reproducible linear-regression portfolio project using scikit-learn's California Housing dataset.

## What it demonstrates

- Data loading and numeric cleaning with pandas
- EDA: summary statistics, missing-value counts, distributions, and target correlations
- Train/test split and median imputation inside a scikit-learn `Pipeline`
- `LinearRegression` training and held-out MAE, RMSE, and R² evaluation
- Actual-vs-predicted, residual, correlation, distribution, and coefficient charts
- Model persistence, a guided Jupyter notebook, and a Marp slide deck

## Run it

```powershell
python -m pip install -r requirements.txt
python house_price_predictor.py
python slides/generate_pdf.py
python reports/generate_pdf_report.py
python predict_ui.py
jupyter notebook notebooks/house_price_predictor.ipynb
```

The first run may download the public California Housing dataset through scikit-learn.

## Outputs

- `artifacts/california_housing_linear_regression.joblib` — fitted preprocessing and model pipeline
- `reports/metrics.json` and `reports/model_report.md` — evaluation results and concise report
- `reports/predictions.csv` — held-out actual values, predictions, and residuals
- `reports/summary_statistics.csv` and `reports/missing_values.csv` — EDA tables
- `reports/california_housing_model_report.pdf` — three-page PDF report
- `reports/figures/` — EDA and diagnostic charts
- `notebooks/house_price_predictor.ipynb` — guided analysis
- `notebooks/task1_ml_linear_regression.ipynb` — submission notebook with executed plots
- `slides/presentation.md` — short Marp presentation
- `slides/california_housing_presentation.pdf` — ready-to-share PDF slide deck
- `predict_ui.py` — desktop form for scoring new dataset-format inputs

## Dataset and responsible use

The target, `MedHouseVal`, is the median owner-occupied value in $100,000s of 1990 USD. Features are aggregated at California block-group level, so this model is an educational baseline only—not a real-estate valuation tool.
