"""Small desktop form for scoring one California Housing input row."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "artifacts" / "california_housing_linear_regression.joblib")
FEATURES = list(MODEL.feature_names_in_)
DEFAULTS = MODEL.named_steps["imputer"].statistics_


def predict() -> None:
    try:
        values = {feature: float(entries[feature].get().strip()) for feature in FEATURES}
    except ValueError:
        messagebox.showerror("Invalid input", "Enter a number in every field.")
        return
    predicted_100k = float(MODEL.predict(pd.DataFrame([values]))[0])
    result.set(f"Predicted median value: ${predicted_100k * 100_000:,.0f} (1990 USD)")


root = tk.Tk()
root.title("California House Price Predictor")
root.resizable(False, False)
frame = ttk.Frame(root, padding=18)
frame.grid()
ttk.Label(frame, text="California House Price Predictor", font=("Segoe UI", 15, "bold")).grid(column=0, row=0, columnspan=2, pady=(0, 6))
ttk.Label(frame, text="Enter the eight California Housing dataset features.").grid(column=0, row=1, columnspan=2, pady=(0, 12))
entries: dict[str, ttk.Entry] = {}
for row, (feature, default) in enumerate(zip(FEATURES, DEFAULTS), start=2):
    ttk.Label(frame, text=feature).grid(column=0, row=row, sticky="w", padx=(0, 12), pady=3)
    entry = ttk.Entry(frame, width=22)
    entry.insert(0, f"{default:.4f}")
    entry.grid(column=1, row=row, pady=3)
    entries[feature] = entry
result = tk.StringVar(value="Enter values and select Predict Price.")
ttk.Button(frame, text="Predict Price", command=predict).grid(column=0, row=10, columnspan=2, pady=(14, 8))
ttk.Label(frame, textvariable=result, wraplength=350, font=("Segoe UI", 10, "bold")).grid(column=0, row=11, columnspan=2)
ttk.Label(frame, text="Educational dataset only; not for real property valuations.", foreground="#666666").grid(column=0, row=12, columnspan=2, pady=(12, 0))
root.mainloop()
