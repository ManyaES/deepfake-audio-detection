import os
import joblib
import pandas as pd

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR, "deepfake_audio_model_improved.pkl"
)

dataset_path = os.path.join(
    BASE_DIR, "ASVspoof2019_test_mfcc.parquet"
)

model = joblib.load(model_path)

# Load the test dataset
test_df = pd.read_parquet(dataset_path)

feature_columns = [f"mfcc_{i}" for i in range(40)]

X_test = test_df[feature_columns]

y_test = (
    test_df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .to_numpy()
)
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Use the model and X_test already loaded
THRESHOLD = 0.50

# Get predictions for evaluation
classes = list(model.classes_)
fake_index = classes.index("fake")
real_index = classes.index("real")

all_probabilities = model.predict_proba(X_test)
fake_probabilities = all_probabilities[:, fake_index]

all_predictions = np.where(
    fake_probabilities >= THRESHOLD,
    "fake",
    "real"
)

accuracy = accuracy_score(y_test, all_predictions)

cm = confusion_matrix(
    y_test,
    all_predictions,
    labels=["fake", "real"]
)

print(f"Model ready. Test accuracy: {accuracy * 100:.2f}%")

# ------------------------------------------------
# Prediction probability chart
# ------------------------------------------------

def make_probability_chart(fake_prob, real_prob):
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.barh(
        ["Fake probability", "Real probability"],
        [fake_prob * 100, real_prob * 100],
        color=["#8977E8", "#37A88B"]
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Estimated probability (%)")
    ax.set_title("Prediction Probability Analysis")
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    return fig

# ------------------------------------------------
# MFCC heatmap
# ------------------------------------------------

def make_heatmap(index):
    values = X_test.iloc[int(index)].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 2.5))

    heatmap = ax.imshow(
        values.reshape(1, -1),
        aspect="auto",
        cmap="viridis"
    )

    ax.set_title(f"MFCC Features — Sample {int(index)}")
    ax.set_xlabel("MFCC feature index")
    ax.set_yticks([])
    ax.set_xticks(range(0, 40, 2))

    fig.colorbar(heatmap, ax=ax, label="Feature value")
    plt.tight_layout()

    return fig

# ------------------------------------------------
# Sample detection
# ------------------------------------------------

def detect_sample(index):
    index = int(index)

    sample = X_test.iloc[[index]]
    probabilities = model.predict_proba(sample)[0]

    fake_prob = float(probabilities[fake_index])
    real_prob = float(probabilities[real_index])

    prediction = "FAKE" if fake_prob >= THRESHOLD else "REAL"
    confidence = max(fake_prob, real_prob) * 100

    if prediction == "FAKE":
        background = "#F0EBFF"
        symbol = "⚠️"
    else:
        background = "#E3F7ED"
        symbol = "✓"

    result = f"""
    <div style="background:{background}; padding:24px;
                border-radius:16px; text-align:center;
                border:1px solid #D7E3F0;">
        <p style="color:#43516B; font-weight:bold;
                  letter-spacing:2px;">DETECTION RESULT</p>
        <h1 style="color:#3047A5;">{symbol} {prediction} AUDIO</h1>
        <p style="color:#34445F;">
            The model classified this sample as likely {prediction.lower()}.
        </p>
        <h3 style="color:#26334D;">
            Predicted-class probability: {confidence:.2f}%
        </h3>
    </div>
    """

    details = f"""
### Sample analysis

- **Sample number:** {index}
- **Model:** Logistic Regression
- **Input features:** 40 MFCC features
- **Threshold:** {THRESHOLD:.2f}
- **Fake probability:** {fake_prob * 100:.2f}%
- **Real probability:** {real_prob * 100:.2f}%
- **Prediction:** {prediction}

The original dataset label is hidden. This is a model prediction,
not a guarantee that the sample is genuinely fake or real.
"""

    return (
        result,
        make_probability_chart(fake_prob, real_prob),
        make_heatmap(index),
        details
    )

# ------------------------------------------------
# Random sample selector
# ------------------------------------------------

def choose_sample(category):
    if category == "Random sample":
        return int(np.random.randint(0, len(X_test)))

    if category == "Sample predicted FAKE":
        indices = np.where(fake_probabilities >= THRESHOLD)[0]
    else:
        indices = np.where(fake_probabilities < THRESHOLD)[0]

    return int(np.random.choice(indices))

# ------------------------------------------------
# Workflow diagram
# ------------------------------------------------

def workflow_diagram():
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        (0.2, "Dataset\nsample", "#E4EDFF"),
        (2.4, "40 MFCC\nfeatures", "#EDE6FF"),
        (4.6, "Standard\nScaler", "#DDF7EC"),
        (6.8, "Logistic\nRegression", "#FFF0D9"),
        (9.0, "FAKE / REAL\nprediction", "#FFE6EF")
    ]

    for x, label, color in stages:
        ax.text(
            x + 0.8, 1.6, label,
            ha="center", va="center",
            color="#26334D", fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor=color,
                edgecolor="#D5DEEC"
            )
        )

    for x in [2.0, 4.2, 6.4, 8.6]:
        ax.annotate(
            "",
            xy=(x + 0.25, 1.6),
            xytext=(x, 1.6),
            arrowprops=dict(arrowstyle="->", color="#64748B")
        )

    ax.text(
        5.5, 0.35,
        "Feature-based classification pipeline",
        ha="center", color="#475569"
    )

    plt.tight_layout()
    return fig

# ------------------------------------------------
# Light-themed interface
# ------------------------------------------------

css = """
.gradio-container {
    max-width: 1250px !important;
    background: #F4F7FC !important;
    color: #26334D !important;
    font-family: Arial, sans-serif !important;
}

#hero {
    background: linear-gradient(120deg,#E5EDFF,#F2E8FF);
    border: 1px solid #D7E0F3;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
}

#hero h1, .gradio-container h2, .gradio-container h3 {
    color: #3047A5 !important;
}

.gradio-container p,
.gradio-container label,
.gradio-container strong,
.gradio-container li {
    color: #26334D !important;
}

.card {
    background: #FFFFFF !important;
    color: #26334D !important;
    border: 1px solid #DCE4F0 !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

button.primary {
    background: #4F63CC !important;
    color: white !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate"
    ),
    css=css,
    title="Deepfake Audio Detection"
) as demo:

    with gr.Column(elem_id="hero"):
        gr.Markdown("""
        # 🎙️ Deepfake Audio Detection

        ### Machine Learning-Based Authenticity Analysis

        Explore model predictions and MFCC feature visualisations.
        """)

    with gr.Row():
        gr.Markdown(f"""
        <div class="card">
        <h3>🧠 Model</h3>
        <h2>Logistic Regression</h2>
        <p>StandardScaler + balanced class weights</p>
        </div>
        """)

        gr.Markdown(f"""
        <div class="card">
        <h3>📊 Test Accuracy</h3>
        <h2>{accuracy * 100:.2f}%</h2>
        <p>ASVspoof2019 test split</p>
        </div>
        """)

        gr.Markdown("""
        <div class="card">
        <h3>🔬 Features</h3>
        <h2>40 MFCCs</h2>
        <p>Precomputed numerical features</p>
        </div>
        """)

        gr.Markdown(f"""
        <div class="card">
        <h3>🎯 Threshold</h3>
        <h2>{THRESHOLD:.2f}</h2>
        <p>Selected using validation data</p>
        </div>
        """)

    gr.Markdown("## 🔍 Interactive Detection")

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            slider = gr.Slider(
                minimum=0,
                maximum=len(X_test) - 1,
                value=100,
                step=1,
                label="Select test sample number"
            )

            category = gr.Dropdown(
                choices=[
                    "Random sample",
                    "Sample predicted FAKE",
                    "Sample predicted REAL"
                ],
                value="Random sample",
                label="Sample selection"
            )

            random_button = gr.Button(
                "🎲 Choose Sample",
                variant="secondary"
            )

            analyse_button = gr.Button(
                "🔍 Analyse Sample",
                variant="primary"
            )

        with gr.Column(scale=2, elem_classes=["card"]):
            result_output = gr.HTML()
            probability_output = gr.Plot(
                label="Prediction probabilities"
            )

    with gr.Column(elem_classes=["card"]):
        gr.Markdown("## 🎨 MFCC Feature Visualisation")
        gr.Markdown(
            "This heatmap shows the sample's 40 numerical features. "
            "It is not an audio recording."
        )
        heatmap_output = gr.Plot(label="MFCC heatmap")
        details_output = gr.Markdown()

    with gr.Column(elem_classes=["card"]):
        gr.Markdown("## ⚙️ How Does It Work?")
        gr.Plot(value=workflow_diagram(), label="Detection workflow")

        gr.Markdown("""
        1. **Dataset:** Select a sample.
        2. **MFCC features:** Use 40 precomputed numerical features.
        3. **StandardScaler:** Scale the input features.
        4. **Logistic Regression:** Estimate fake and real probabilities.
        5. **Prediction:** Fake probability ≥ 0.50 gives FAKE;
           otherwise, the prediction is REAL.
        """)

    with gr.Column(elem_classes=["card"]):
        gr.Markdown("## 📈 Model Evaluation")

        cm_fig, cm_ax = plt.subplots(figsize=(6, 4))
        cm_ax.imshow(cm, cmap="Blues")
        cm_ax.set_xticks([0, 1])
        cm_ax.set_yticks([0, 1])
        cm_ax.set_xticklabels(["Predicted Fake", "Predicted Real"])
        cm_ax.set_yticklabels(["Actual Fake", "Actual Real"])
        cm_ax.set_title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                cm_ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", color="black"
                )

        plt.tight_layout()
        gr.Plot(value=cm_fig, label="Test confusion matrix")

        gr.Markdown("""
        **Limitation:** This application analyses precomputed MFCC
        features. It does not upload or play original audio files.
        A prediction is an estimate, not proof of authenticity.
        """)

    analyse_button.click(
        fn=detect_sample,
        inputs=[slider],
        outputs=[
            result_output,
            probability_output,
            heatmap_output,
            details_output
        ]
    )

    random_button.click(
        fn=choose_sample,
        inputs=[category],
        outputs=[slider]
    )

    demo.load(
        fn=detect_sample,
        inputs=[slider],
        outputs=[
            result_output,
            probability_output,
            heatmap_output,
            details_output
        ]
    )

import os

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
