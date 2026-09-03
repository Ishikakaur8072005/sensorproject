from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sensor Fault Detection</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; }
        .container { max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h2 { color: #333; }
        .btn { display: inline-block; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; text-decoration: none; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        form { margin-top: 20px; }
        input[type="file"] { margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Sensor Fault Detection Pipeline</h2>
        <p>Trigger model training or upload sensor data CSV for fault prediction.</p>
        
        <form action="/train" method="post">
            <button type="submit" class="btn">Start Training Pipeline</button>
        </form>
        
        <hr style="margin: 30px 0;">

        <h3>Upload Wafer Data CSV for Prediction</h3>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required><br>
            <button type="submit" class="btn">Predict Faults</button>
        </form>
    </div>
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
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                df = pd.read_csv(file)
                predictions = pipeline.predict(df)
            else:
                return jsonify({"status": "error", "message": "No file uploaded"}), 400
            
            df["Prediction"] = predictions
            df["Prediction"] = df["Prediction"].map({0: "Bad / Faulty", 1: "Good"})
            return jsonify({
                "status": "success",
                "total_records": len(df),
                "predictions_summary": df["Prediction"].value_counts().to_dict(),
                "predictions": df["Prediction"].tolist()[:50]
            })
        else:
            return render_template_string(HTML_TEMPLATE)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
