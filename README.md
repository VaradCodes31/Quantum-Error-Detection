# 🛡️ Quantum Error Detection & XAI Dashboard

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Qiskit](https://img.shields.io/badge/Quantum-Qiskit-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, physics-informed machine learning system designed to classify and explain quantum noise channels in single-qubit circuits. This project bridges the gap between raw quantum simulation data and human-readable diagnostics using a high-performance **5-Model Ensemble** and **Explainable AI (XAI)**.

---

## 🚀 Overview

Quantum computers are inherently noisy. Identifying whether an error stems from **Bit Flips**, **Phase Flips**, **Measurement (Readout)**, or **Depolarization** is critical for building robust quantum algorithms.

### Key Capabilities:
- **Ensemble Intelligence**: Combines MLP, Random Forest, Extra Trees, Hist-Gradient Boosting, and Gradient Boosting for 41%+ accuracy on complex stochastic data.
- **Physics-Informed Features**: Engineered features based on theoretical quantum likelihoods ($1 - (1-s)^n$).
- **Live XAI Dashboard**: Real-time "Logic Breakdowns" explaining *why* a specific error was predicted.
- **Global Analytics**: Integrated Confusion Matrix and Cross-Model benchmarking.

---

## 🛠️ Technical Stack

- **Quantum Backend**: `Qiskit Aer` (High-fidelity noise simulation).
- **ML Engine**: `Scikit-Learn` (VotingClassifier with soft-voting weights).
- **Deep Learning**: `MLPClassifier` (Multi-layer Perceptron for non-linear mapping).
- **Visualization**: `Streamlit`, `Seaborn`, `Matplotlib`.
- **Explainability**: `SHAP` & Custom Physical Heuristic Logic trees.

---

## 📂 Project Structure

```text
.
├── app.py                # Main Streamlit Dashboard
├── final_train.py        # Production-grade Ensemble training
├── generate_confusion_matrix.py # Evaluation & Metrics generation
├── src/
│   ├── preprocessing.py  # Physics-informed feature engineering
│   ├── ensemble.py       # Quantum model architecture
│   └── classical_models.py # Scikit-learn estimator definitions
├── data/                 # Dataset storage (20k samples)
└── results/              # Model artifacts, scalers, and plots
```

---

## ⚡ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/VaradCodes31/Quantum-Error-Detection.git
cd DL_CCA

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Models
```bash
# Generate the 20k sample dataset
python3 src/error_detection_dataset_generator.py

# Train the unified 5-model ensemble
python3 final_train.py

# Generate performance metrics (Confusion Matrix)
python3 generate_confusion_matrix.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📊 Model Performance

The system uses a **Soft-Voting Ensemble** where each model contributes to the final decision. 

| Error Type | Precision | Recall | Physical Signature |
| :--- | :--- | :--- | :--- |
| **Readout** | 63% | 63% | Independent of gate count; purely measurement-based. |
| **Bit Flip** | 37% | 38% | Cumulative X-axis noise building with circuit depth. |
| **Phase Flip**| 28% | 25% | Z-axis noise; visible only with Hadamard basis-shifts. |
| **Depolarizing**| 35% | 38% | Uniform randomization across X, Y, and Z. |

### 🎯 Global Confusion Matrix
The dashboard includes a normalized confusion matrix highlighting the statistical overlaps inherent in single-qubit noise characterization.

---

## 📖 How It Works: The "Logic Breakdown"

When you input circuit parameters (Gate Count, Depth, Noise Strength) into the dashboard, the system:
1. **Preprocesses** the data using `RobustScaler`.
2. **Predicts** the error type using the weighted ensemble.
3. **Explains** the result by mapping feature importance (from the Random Forest lead) to physical quantum principles. 

*Example Reasoning: "Bit Flip is predicted because the error rate is significantly higher than the baseline noise, indicating cumulative gate-level decay."*

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
