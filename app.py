import os
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils.main_utils import MainUtils
from src.utils.shap_explainer import explain_row, build_background_sample
from src.constant import artifact_folder

app = Flask(__name__)
utils = MainUtils()

MODEL_PATH = os.path.join(artifact_folder, "model.pkl")
PREPROCESSOR_PATH = os.path.join(artifact_folder, "preprocessor.pkl")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wafer Fault Detection</title>
<style>
  :root {
    --bg: #15181B;
    --panel: #1D2024;
    --panel-border: #2A2E33;
    --text: #E8E6E1;
    --text-dim: #8B9096;
    --pass: #4FAE9E;
    --fault: #D98E3E;
    --fault-bright: #E8A33D;
    --mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    padding: 48px 24px 80px;
  }
  .wrap { max-width: 780px; margin: 0 auto; }

  .masthead { margin-bottom: 40px; }
  .masthead .tag {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
    letter-spacing: 0.02em;
  }
  .masthead h1 { font-size: 28px; font-weight: 600; margin: 6px 0 0; letter-spacing: -0.01em; }
  .masthead p { color: var(--text-dim); font-size: 14px; margin: 8px 0 0; max-width: 540px; line-height: 1.5; }

  .panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 24px 28px;
    margin-bottom: 20px;
  }
  .panel h2 { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
  .panel .desc { color: var(--text-dim); font-size: 13px; margin: 0 0 18px; line-height: 1.5; }

  button, .file-btn {
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 600;
    padding: 9px 18px;
    border-radius: 4px;
    border: 1px solid var(--panel-border);
    background: #262A2F;
    color: var(--text);
    cursor: pointer;
    transition: border-color 0.15s ease;
  }
  button:hover, .file-btn:hover { border-color: #454B52; }
  button:disabled { opacity: 0.5; cursor: default; }
  button.small { padding: 4px 10px; font-size: 11px; font-weight: 500; }

  input[type="file"] { display: none; }
  .file-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .file-name { font-family: var(--mono); font-size: 12px; color: var(--text-dim); }

  .result { margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--panel-border); display: none; }
  .result.show { display: block; }

  .status-line {
    font-family: var(--mono); font-size: 12px;
    display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .dot.ok { background: var(--pass); }
  .dot.err { background: var(--fault-bright); }
  .dot.pending { background: var(--text-dim); animation: pulse 1.2s infinite ease-in-out; }
  @keyframes pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }

  .metric-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px;
    background: var(--panel-border); border: 1px solid var(--panel-border);
    border-radius: 4px; overflow: hidden;
  }
  .metric { background: var(--panel); padding: 14px 16px; }
  .metric .label { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
  .metric .value { font-family: var(--mono); font-size: 20px; font-weight: 600; }
  .metric .value.pass { color: var(--pass); }
  .metric .value.fault { color: var(--fault-bright); }

  .pred-list {
    font-family: var(--mono); font-size: 12px; max-height: 320px; overflow-y: auto;
    border: 1px solid var(--panel-border); border-radius: 4px; margin-top: 14px;
  }
  .pred-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 12px; border-bottom: 1px solid var(--panel-border);
  }
  .pred-row:last-child { border-bottom: none; }
  .pred-row .tag-fault { color: var(--fault-bright); }
  .pred-row .tag-pass { color: var(--pass); }
  .pred-row .right { display: flex; align-items: center; gap: 10px; }

  .explain-box {
    grid-column: 1 / -1;
    padding: 14px 12px;
    border-bottom: 1px solid var(--panel-border);
    background: #17191C;
  }
  .explain-box img { width: 100%; border-radius: 4px; display: block; }
  .explain-note {
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
    margin-top: 8px; line-height: 1.5;
  }

  .error-box {
    font-family: var(--mono); font-size: 12px; color: var(--fault-bright);
    background: rgba(217, 142, 62, 0.08); border: 1px solid rgba(217, 142, 62, 0.3);
    border-radius: 4px; padding: 10px 14px; white-space: pre-wrap;
  }

  footer { max-width: 780px; margin: 40px auto 0; font-family: var(--mono); font-size: 11px; color: var(--text-dim); }
</style>
</head>
<body>
<div class="wrap">

  <div class="masthead">
    <div class="tag">UCI SECOM &middot; EasyEnsembleClassifier</div>
    <h1>Wafer Fault Detection</h1>
    <p>Runs the sensor-data pipeline against a classifier tuned for a rare-fault, high-dimensional dataset. Trigger a retrain, upload wafer readings to flag likely faults, and explain any flagged wafer's prediction.</p>
  </div>

  <div class="panel">
    <h2>Retrain model</h2>
    <p class="desc">Re-runs ingestion, transformation, and training end to end using the current dataset.</p>
    <button id="trainBtn" onclick="runTrain()">Start training pipeline</button>
    <div class="result" id="trainResult"></div>
  </div>

  <div class="panel">
    <h2>Predict from CSV</h2>
    <p class="desc">Upload wafer sensor readings (same feature columns as training, no target column). Flagged faults can be explained individually below.</p>
    <div class="file-row">
      <label class="file-btn" for="fileInput">Choose CSV</label>
      <input type="file" id="fileInput" accept=".csv" onchange="fileChosen()">
      <span class="file-name" id="fileName">No file selected</span>
      <button id="predictBtn" onclick="runPredict()" disabled>Predict faults</button>
    </div>
    <div class="result" id="predictResult"></div>
  </div>

</div>
<footer>sensor_project &middot; local dev server</footer>

<script>
let uploadedFile = null;

function fileChosen() {
  const input = document.getElementById('fileInput');
  uploadedFile = input.files.length ? input.files[0] : null;
  document.getElementById('fileName').textContent = uploadedFile ? uploadedFile.name : 'No file selected';
  document.getElementById('predictBtn').disabled = !uploadedFile;
}

async function runTrain() {
  const btn = document.getElementById('trainBtn');
  const box = document.getElementById('trainResult');
  btn.disabled = true;
  box.className = 'result show';
  box.innerHTML = '<div class="status-line"><span class="dot pending"></span>Training in progress&hellip; this can take a moment.</div>';

  try {
    const res = await fetch('/train', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      box.innerHTML = `
        <div class="status-line"><span class="dot ok"></span>Training completed</div>
        <div class="error-box" style="color: var(--text-dim); border-color: var(--panel-border); background: transparent;">Saved model: ${data.model_path || 'artifacts/model.pkl'}</div>
      `;
    } else {
      box.innerHTML = `<div class="status-line"><span class="dot err"></span>Training failed</div><div class="error-box">${data.message}</div>`;
    }
  } catch (err) {
    box.innerHTML = `<div class="status-line"><span class="dot err"></span>Request failed</div><div class="error-box">${err}</div>`;
  } finally {
    btn.disabled = false;
  }
}

async function runPredict() {
  const btn = document.getElementById('predictBtn');
  const box = document.getElementById('predictResult');
  if (!uploadedFile) return;

  btn.disabled = true;
  box.className = 'result show';
  box.innerHTML = '<div class="status-line"><span class="dot pending"></span>Scoring wafers&hellip;</div>';

  const formData = new FormData();
  formData.append('file', uploadedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'success') {
      const good = data.predictions_summary['Good'] || 0;
      const bad = data.predictions_summary['Bad / Faulty'] || 0;
      const rows = data.predictions.map((p, i) => {
        const isFault = p === 'Bad / Faulty';
        return `
        <div id="row-${i}">
          <div class="pred-row">
            <span>Row ${i + 1}</span>
            <span class="right">
              <span class="${isFault ? 'tag-fault' : 'tag-pass'}">${p}</span>
              ${isFault ? `<button class="small" onclick="explainRow(${i})" id="explainBtn-${i}">Explain</button>` : ''}
            </span>
          </div>
          <div id="explain-${i}"></div>
        </div>`;
      }).join('');
      box.innerHTML = `
        <div class="status-line"><span class="dot ok"></span>Scored ${data.total_records} records</div>
        <div class="metric-grid">
          <div class="metric"><div class="label">Flagged as good</div><div class="value pass">${good}</div></div>
          <div class="metric"><div class="label">Flagged as faulty</div><div class="value fault">${bad}</div></div>
        </div>
        <div class="pred-list">${rows}</div>
      `;
    } else {
      box.innerHTML = `<div class="status-line"><span class="dot err"></span>Prediction failed</div><div class="error-box">${data.message}</div>`;
    }
  } catch (err) {
    box.innerHTML = `<div class="status-line"><span class="dot err"></span>Request failed</div><div class="error-box">${err}</div>`;
  } finally {
    btn.disabled = false;
  }
}

async function explainRow(rowIndex) {
  const target = document.getElementById(`explain-${rowIndex}`);
  const btn = document.getElementById(`explainBtn-${rowIndex}`);
  btn.disabled = true;
  target.innerHTML = `<div class="explain-box"><div class="status-line"><span class="dot pending"></span>Computing SHAP explanation&hellip; ~10s</div></div>`;

  const formData = new FormData();
  formData.append('file', uploadedFile);
  formData.append('row_index', rowIndex);

  try {
    const res = await fetch('/explain', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'success') {
      target.innerHTML = `
        <div class="explain-box">
          <img src="data:image/png;base64,${data.image}" alt="SHAP explanation">
          <div class="explain-note">Top sensors pushing this prediction toward "Bad / Faulty." Bar length = magnitude of that sensor's contribution.</div>
        </div>`;
    } else {
      target.innerHTML = `<div class="explain-box"><div class="error-box">${data.message}</div></div>`;
    }
  } catch (err) {
    target.innerHTML = `<div class="explain-box"><div class="error-box">${err}</div></div>`;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/train", methods=["GET", "POST"])
def train():
    try:
        pipeline = TrainPipeline()
        model_path = pipeline.run_pipeline()
        return jsonify({"status": "success", "message": "Training completed successfully", "model_path": model_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        if request.method == "POST":
            pipeline = PredictionPipeline()
            if "file" in request.files and request.files["file"].filename != "":
                file = request.files["file"]
                df = pd.read_csv(file)
                predictions = pipeline.predict(df)
            else:
                return jsonify({"status": "error", "message": "No file uploaded"}), 400

            df["Prediction"] = predictions
            df["Prediction"] = df["Prediction"].map({0: "Good", 1: "Bad / Faulty"})
            return jsonify({
                "status": "success",
                "total_records": len(df),
                "predictions_summary": df["Prediction"].value_counts().to_dict(),
                "predictions": df["Prediction"].tolist()[:50],
            })
        else:
            return render_template_string(HTML_TEMPLATE)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """Explains a single row's prediction using SHAP, returns a base64 PNG bar chart."""
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        row_index = int(request.form.get("row_index", 0))

        file = request.files["file"]
        df = pd.read_csv(file)

        drop_cols = ["_id", "id", "Unnamed: 0", "Good/Bad", "quality"]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        if row_index >= len(df):
            return jsonify({"status": "error", "message": "Row index out of range"}), 400

        model = utils.load_object(MODEL_PATH)
        preprocessor = utils.load_object(PREPROCESSOR_PATH)

        feature_names = df.columns.tolist()
        X_scaled_full = preprocessor.transform(df)
        background = build_background_sample(X_scaled_full, n=15)

        row_df = df.iloc[[row_index]]
        image_b64 = explain_row(model, preprocessor, row_df, feature_names, background)

        return jsonify({"status": "success", "image": image_b64})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)