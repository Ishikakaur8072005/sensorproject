import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# --- dashboard theme, matches app.py's CSS variables ---
BG = "#1D2024"
TEXT = "#E8E6E1"
DIM = "#8B9096"
FAULT = "#E8A33D"
GRID = "#2A2E33"

_STYLE = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": DIM,
    "ytick.color": TEXT,
    "axes.edgecolor": GRID,
    "font.family": "monospace",
    "font.size": 10,
}


def explain_row(model, preprocessor, raw_row_df, feature_names, background_scaled, max_display=8):
    """
    Explains a single row's prediction with SHAP, returns a base64 PNG
    (bar chart of the top contributing sensors toward the 'Faulty' class).

    model:              the trained classifier (must expose predict_proba)
    preprocessor:       fitted sklearn preprocessor (imputer + scaler pipeline)
    raw_row_df:         a 1-row DataFrame of raw (unscaled) sensor readings
    feature_names:      list of feature column names, in order
    background_scaled:  a small (~15-row) sample of already-scaled data used
                         as the SHAP reference distribution
    """
    scaled_row = preprocessor.transform(raw_row_df)

    explainer = shap.Explainer(model.predict_proba, background_scaled, feature_names=feature_names)
    n_features = len(feature_names)
    shap_values = explainer(scaled_row, max_evals=2 * n_features + 1)

    # class index 1 = "Faulty" in this project's label encoding
    sv_fault = shap_values[..., 1]

    with plt.rc_context(_STYLE):
        plt.figure(figsize=(7, 4.2))
        shap.plots.bar(sv_fault[0], max_display=max_display, show=False)

        ax = plt.gca()
        for bar in ax.patches:
            bar.set_facecolor(FAULT)
            bar.set_edgecolor(FAULT)
        for text in ax.texts:
            text.set_color(FAULT)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, facecolor=BG)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


def build_background_sample(X_scaled, n=15, random_state=42):
    """Small reference sample SHAP needs to estimate feature impact."""
    return shap.sample(X_scaled, n, random_state=random_state)