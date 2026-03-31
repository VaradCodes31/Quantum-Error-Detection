# Quantum Error Detection & XAI Project Documentation

## 1. Project Overview
This project implements a physics-informed machine learning system designed to classify quantum noise in single-qubit circuits. By bridging the gap between quantum simulation and deep learning, the system provides both accurate predictions and human-readable explanations via an interactive dashboard.

### Core Objectives
*   **Classification**: Distinguish between four fundamental quantum noise channels.
*   **Transparency**: Provide a "Logic Breakdown" that maps model decisions back to quantum physical principles.
*   **Efficiency**: Optimized for low-latency training and inference on consumer-grade Apple Silicon hardware.

---

## 2. Technical Stack
*   **Quantum Backend**: `Qiskit Aer` (High-performance simulation of noisy circuits).
*   **Ensemble Core**: `Scikit-Learn` (VotingClassifier with 5 architectures).
*   **Neural Network**: `MLPClassifier` (Multi-layer Perceptron for non-linear feature mapping).
*   **Dashboard**: `Streamlit` (Interactive visualization and real-time inference).
*   **Explainability**: `SHAP` & Custom Heuristic Logic trees.

---

## 3. The Technical Pipeline

The system follows a strict, end-to-end data processing pipeline ensuring that physical quantum states are accurately mapped into machine-learnable features.

1.  **Simulation Layer (`Qiskit`)**: 
    - Random circuits are generated with depths $d \in [5, 30]$ and random gate sequences $(H, X, Z, RZ, SX)$.
    - Noise models are injected (BitFlip, PhaseFlip, Readout, Depolarizing).
    - Circuits are executed with 1024 shots to produce stochastic counts.
2.  **Feature Integration Layer (`src/preprocessing.py`)**:
    - Raw counts are converted to proportions.
    - **Theoretical Likelihoods** are calculated using quantum mechanics formulas ($1 - (1-s)^n$) to provide the models with a "physics-hint" of the expected error.
3.  **Normalization Layer (`RobustScaler`)**:
    - All 21 features are scaled based on their interquartile range. This is critical for the Multi-Layer Perceptron (neural network) to converge on high-variance quantum count data.
4.  **Inference Layer (`VotingClassifier`)**:
    - The processed features are passed to 5 parallel models simultaneously. Each model contributes its unique mathematical perspective to the final probability vector.
5.  **Explainability Layer (`app.py`)**:
    - Prediction results are passed to the XAI engine, which uses the lead model's (Random Forest) feature attribution to explain the prediction in plain English.

---

## 4. Dataset & Mathematical Simulation

### 4.1 Dataset Properties
The dataset consists of 20,000+ generated samples, ensuring a balanced distribution across all 4 noise classes.
*   **Input Features**:
    *   `prepared_state`: Initial state ($|0\rangle$ or $|1\rangle$).
    *   `measured_counts`: Stochastic 0/1 outcomes from 1024 shots.
    *   `gate_count` & `circuit_depth`: Metrics of total circuit complexity.
    *   `noise_strength`: Underlying error probability per operation $(s \in [0.01, 0.45])$.
    *   `has_h_gate`: Boolean flag indicating Hadamard usage (translates Z-errors into the computational basis).

### 4.2 Noise Modeling Details
*   **Readout Error**: Stochastic flipping of the final measurement result.
*   **Bit Flip (X)**: Cumulative Pauli-X noise that grows with `circuit_depth`.
*   **Phase Flip (Z)**: Cumulative Pauli-Z noise. Only visible if an $H$-gate is present.
*   **Depolarizing**: A generalized channel where the qubit state is uniformly randomized across $X, Y, \text{and } Z$ axes.

---

## 5. Estimator Mechanics: How the Models Work

The system uses a **Heterogeneous Ensemble** to leverage the strengths of different learning paradigms.

### 5.1 Estimator Breakdown
| Model | Configuration | Internal Mechanism |
| :--- | :--- | :--- |
| **MLP (Perceptron)** | $256 \times 128 \times 64$ layers | A **Neural Network** that uses backpropagation to learn non-linear decision boundaries between depth and error rate. |
| **Random Forest** | 500 Trees, Depth 25 | A **Bagging Ensemble** that builds multiple decision trees on random data subsets and averages them to minimize variance. |
| **Extra Trees** | 500 Trees, Depth 25 | Similar to Random Forest but chooses split points **randomly** rather than optimally, making it more robust to noisy quantum data. |
| **HGB (Hist-based)** | Max Iter: 300 | A **Gradient Boosting** model that bins continuous features into discrete histograms, significantly accelerating the training on the 20k dataset. |
| **Gradient Boosting** | 300 Est., Depth 6 | An iterative ensemble that focuses on correcting the specific classification errors made by the previous trees in the sequence. |

---

## 6. The Voting Mechanism
Instead of a single "Winner-Takes-All" prediction, we use **Soft Voting**:
1.  Each of the 5 models calculates a probability vector $[P_{\text{Readout}}, P_{\text{Bit}}, P_{\text{Phase}}, P_{\text{Depolarize}}]$.
2.  **Weighting**: We assign **higher weights (4.0 and 3.0)** to the tree-based models (RF and ET) because audit tests showed they are superior at resolving the specific "Bit Flip vs. Readout" overlap at low gate counts.
3.  The final result is the **argmax** of the weighted average across all 5 architectures.

---

## 7. Execution Guide
1.  **Generate Dataset**: `python src/error_detection_dataset_generator.py` (Creates 20k sample CSV).
2.  **Train Ensemble**: `python final_train.py` (Generates the `.joblib` artifacts).
3.  **Launch UI**: `streamlit run app.py`.
4.  **Venv Actication**: `source /Users/admin/Documents/Projects/DL_CCA/venv/bin/activate`

---

### Final Status: Optimized 🛡️🚀
The system provides a robust framework for quantum characterization, ensuring that even as circuits become deeper and noise more complex, the source of the error remains identifiable.
