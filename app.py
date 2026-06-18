"""Project 06 — Credit Card Fraud Detection Demo

Interactive Gradio app for the winning model from the imbalance-strategy
comparison: XGBoost with class weighting (PR-AUC = 0.880, recall@P90 = 0.837).

Two input modes:
  - Simple:   adjust Amount + pick a feature-pattern preset sampled from
              real legitimate / fraud transactions in the training data
  - Advanced: enter all 30 raw feature values directly (V1-V28, Time, Amount)

Run locally:
    python app.py

Deploy to Hugging Face Spaces:
    Push this file + requirements.txt + models/ to a Spaces repo
    (SDK: Gradio). See README for the exact file list.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

MODELS_DIR = Path("models")

model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / "fraud_xgboost_class_weighted.json")

scaler = joblib.load(MODELS_DIR / "scaler.joblib")

feature_columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())
metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())

V_COLUMNS = [c for c in feature_columns if c.startswith("V")]

# --- Preset patterns for Simple mode -----------------------------------
# V1-V28 are PCA-anonymized components with no inherent real-world meaning,
# so a "risk profile" can't be built from first principles. These presets
# are the actual class-conditional feature means computed from the real
# training data at model-training time (see src/train_final_model.py ->
# compute_preset_patterns), not hand-guessed values -- so Simple mode
# reflects genuine legitimate / fraud feature patterns from the dataset.
PRESETS = json.loads((MODELS_DIR / "preset_patterns.json").read_text())

DEFAULT_TIME = 80000.0


def predict_fraud(feature_dict: dict, amount: float, time: float = DEFAULT_TIME) -> dict:
    """Run the model on a single transaction and return a label dict for Gradio."""
    row = {col: feature_dict.get(col, 0.0) for col in V_COLUMNS}
    row["Time"] = time
    row["Amount"] = amount

    X = pd.DataFrame([row])[feature_columns]
    X_scaled = X.copy()
    X_scaled[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    proba_fraud = float(model.predict_proba(X_scaled)[0, 1])
    proba_legit = 1.0 - proba_fraud

    return {"Fraud": proba_fraud, "Legitimate": proba_legit}


def predict_simple(preset_name: str, amount: float):
    feature_dict = PRESETS[preset_name]
    return predict_fraud(feature_dict, amount)


def predict_advanced(amount, time, *v_values):
    feature_dict = dict(zip(V_COLUMNS, v_values))
    return predict_fraud(feature_dict, amount, time)


with gr.Blocks(title="Project 06 — Fraud Detection Demo") as demo:
    gr.Markdown(
        f"""
        # Credit Card Fraud Detection — Live Demo

        **Model:** XGBoost with class weighting
        **PR-AUC:** {metadata['metrics']['pr_auc']:.3f}  |
        **Recall @ 90% precision:** {metadata['metrics']['recall_at_p90']:.3f}  |
        **F1:** {metadata['metrics']['f1']:.3f}

        This model was selected from a 13-run comparison across five class-imbalance
        strategies (baseline, class weighting, SMOTE, random undersampling, and an
        unsupervised Isolation Forest baseline). Full results and methodology:
        [GitHub repo](https://github.com/muhammed-keita-ml/project-06-fraud-detection-pipeline) ·
        [DagsHub experiments](https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline)

        The underlying dataset's V1–V28 features are PCA-anonymized for privacy and have
        no inherent real-world meaning, so this demo offers two ways to explore the model.
        """
    )

    with gr.Tabs():
        with gr.Tab("Simple Mode"):
            gr.Markdown(
                "Pick a feature pattern representative of legitimate or fraudulent "
                "transactions in the training data, and adjust the transaction amount."
            )
            with gr.Row():
                with gr.Column():
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Typical legitimate pattern",
                        label="Transaction Pattern",
                    )
                    amount_simple = gr.Slider(
                        minimum=0, maximum=5000, value=50, step=1,
                        label="Transaction Amount ($)",
                    )
                    btn_simple = gr.Button("Classify Transaction", variant="primary")
                with gr.Column():
                    output_simple = gr.Label(label="Prediction", num_top_classes=2)

            btn_simple.click(
                fn=predict_simple,
                inputs=[preset_dropdown, amount_simple],
                outputs=output_simple,
            )

        with gr.Tab("Advanced Mode"):
            gr.Markdown(
                "Enter raw feature values directly. V1–V28 are PCA components "
                "(typically in the range -5 to 5); Time is seconds since the "
                "first transaction in the dataset."
            )
            with gr.Row():
                amount_adv = gr.Number(value=50.0, label="Amount ($)")
                time_adv = gr.Number(value=DEFAULT_TIME, label="Time (seconds)")

            v_inputs = []
            with gr.Row():
                for i in range(0, 14):
                    v_inputs.append(gr.Number(value=0.0, label=V_COLUMNS[i]))
            with gr.Row():
                for i in range(14, 28):
                    v_inputs.append(gr.Number(value=0.0, label=V_COLUMNS[i]))

            btn_adv = gr.Button("Classify Transaction", variant="primary")
            output_adv = gr.Label(label="Prediction", num_top_classes=2)

            btn_adv.click(
                fn=predict_advanced,
                inputs=[amount_adv, time_adv] + v_inputs,
                outputs=output_adv,
            )

    gr.Markdown(
        """
        ---
        **Note on metrics:** this model is evaluated primarily on PR-AUC and recall at
        fixed precision rather than accuracy, since fraud cases make up only ~0.17% of
        transactions in the training data — accuracy alone would be misleading at this
        level of class imbalance.
        """
    )


if __name__ == "__main__":
    demo.launch()
