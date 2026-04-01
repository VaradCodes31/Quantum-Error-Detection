import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from src.ensemble import QuantumEnsemble
from src.preprocessing import feature_engineering
import matplotlib.pyplot as plt
import shap

# Set page config
st.set_page_config(page_title="Quantum Error Detector", layout="wide")

st.title("🛡️ Quantum Error Detection & XAI Dashboard")
st.markdown("""
This dashboard predicts the type of quantum error based on circuit parameters and noise characteristics.
It uses an **Optimized High-Performance Ensemble** (MLP, RF, ET, HGB, GB).
""")

@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load('results/scaler.pkl')
        encoder = joblib.load('results/encoder.pkl')
        ensemble = joblib.load('results/best_ensemble_model.joblib')
        
        with open('results/feature_names.pkl', 'rb') as f:
            feature_names = joblib.load(f)
            
        # Load benchmarks if they exist
        benchmarks = {}
        if os.path.exists('results/benchmarks.pkl'):
            benchmarks = joblib.load('results/benchmarks.pkl')
            
        return ensemble, scaler, encoder, feature_names, benchmarks
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, {}

ensemble, scaler, encoder, feature_names, benchmarks = load_artifacts()

# --- NEW LARGE BENCHMARKING SECTION ---
if benchmarks:
    with st.expander(" Model Performance Benchmarks (Global Tuning)", expanded=True):
        bench_df = pd.DataFrame(list(benchmarks.items()), columns=['Model', 'Accuracy'])
        bench_df = bench_df.sort_values(by='Accuracy', ascending=False)
        
        # Display as a large, wide horizontal bar chart
        fig_bench, ax_bench = plt.subplots(figsize=(12, 4))
        colors = plt.cm.RdYlGn(bench_df['Accuracy'] / bench_df['Accuracy'].max())
        bars = ax_bench.barh(bench_df['Model'], bench_df['Accuracy'], color=colors)
        ax_bench.set_xlabel('Validation Accuracy')
        ax_bench.set_title('Cross-Model Performance Comparison')
        ax_bench.set_xlim(0, 1.0)
        
        # Add text labels on bars
        for bar in bars:
            width = bar.get_width()
            ax_bench.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{width*100:.1f}%', va='center', fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig_bench)
        
        # Mini stat cards
        cols = st.columns(len(benchmarks))
        for i, (m_name, acc) in enumerate(bench_df.values):
            cols[i].metric(m_name, f"{acc*100:.1f}%")

        # --- ADDED CONFUSION MATRIX ---
        st.markdown("---")
        st.subheader(" Global Confusion Matrix (normalized)")
        if os.path.exists('results/confusion_matrix.png'):
            st.image('results/confusion_matrix.png', use_container_width=True, 
                     caption="Normalized Confusion Matrix on Test Set")
        else:
            st.warning("Confusion matrix image not found. Run `generate_confusion_matrix.py`.")

if ensemble:
    st.sidebar.success("✅ Unified Ensemble Loaded")

    # Sidebar inputs
    st.sidebar.header("Quantum Circuit Parameters")
    
    ps = st.sidebar.selectbox("Prepared State", [0, 1])
    ns = st.sidebar.slider("Noise Strength", 0.01, 0.20, 0.05, step=0.01)
    gc = st.sidebar.slider("Gate Count", 1, 10, 3)
    cd = st.sidebar.slider("Circuit Depth", 1, 20, 10)
    er = st.sidebar.slider("Observed Error Rate", 0.0, 0.5, 0.1, step=0.01)
    has_h = st.sidebar.checkbox("Contains Hadamard (X-Basis) Gates?", value=True)
    
    # STOCHASTIC SIMULATION
    shots = 1024
    m1 = np.random.binomial(shots, er) if ps == 0 else np.random.binomial(shots, 1-er)
    m0 = shots - m1
    
    st.sidebar.info(f"Stochastic Counts (Simulated): 0: {m0}, 1: {m1}")

    st.sidebar.info(f"Stochastic Counts (Simulated): 0: {m0}, 1: {m1}")

    def get_input_df(ps, ns, gc, cd, er, m0, m1, has_h):
        raw_data = {
            'prepared_state': [ps],
            'measured_0': [m0],
            'measured_1': [m1],
            'error_rate': [er],
            'noise_strength': [ns],
            'gate_count': [gc],
            'circuit_depth': [cd],
            'has_h_gate': [1 if has_h else 0]
        }
        df = pd.DataFrame(raw_data)
        # Apply the exact same feature engineering as training
        df_feat = feature_engineering(df)
        X = df_feat.drop(columns=['measured_0', 'measured_1'])
        return X[feature_names]

    input_df = get_input_df(ps, ns, gc, cd, er, m0, m1, has_h)
    input_scaled = scaler.transform(input_df)

    # Prediction
    if st.button("Predict Error Type"):
        # 1. Heuristic Override (Physical certainty)
        heuristic_label = None
        if gc == 0 and er > 0:
            heuristic_label = "readout"
        elif er == 0 and ns > 0 and has_h == False:
            heuristic_label = "phase_flip" # Z-noise on |0> is invisible

        # 2. Model Prediction
        probs = ensemble.predict_proba(input_scaled)
        pred_class = np.argmax(probs, axis=1)[0]
        pred_label = encoder.inverse_transform([pred_class])[0]
        
        # Apply override if applicable
        if heuristic_label:
            pred_label = heuristic_label
            st.sidebar.warning(f"Note: Physical heuristic applied ({heuristic_label})")

        confidence = probs[0][pred_class] * 100

        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"### Predicted Error: **{pred_label.upper()}**")
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Probability Bar Chart
            prob_df = pd.DataFrame({
                'Error Type': encoder.classes_,
                'Probability': probs[0]
            })
            st.bar_chart(prob_df.set_index('Error Type'))

        with col2:
            st.info("### Explainable AI (XAI)")
            
            # Use the Random Forest estimator from the ensemble for XAI
            # ensemble.estimators_ contains [mlp, rf, et, hgb, gb]
            target_model = ensemble.estimators_[1] 
            t_name = "Random Forest (Ensemble Lead)"
            
            st.write(f"Generating Feature Importance using **{t_name}**...")
            
            try:
                # 1. Dynamic + Physical Logic Breakdown
                st.markdown("####  Logic Breakdown")
                
                # Extract key values for logic text
                gc_val = input_df['gate_count'].values[0]
                ns_val = input_df['noise_strength'].values[0]
                er_val = input_df['error_rate'].values[0]
                
                # Get sorted probabilities
                sorted_indices = np.argsort(probs[0])[::-1]
                top_p = probs[0][sorted_indices[0]]
                next_p = probs[0][sorted_indices[1]]
                next_label = encoder.classes_[sorted_indices[1]]
                
                # Feature-based context
                feat_imps = pd.Series(target_model.feature_importances_, index=feature_names).sort_values(ascending=False)
                top_feat = feat_imps.index[0]
                
                # Detailed Physical Map
                physical_map = {
                    "readout": "Physical readout errors occur only during measurement and are independent of gate count. Since your error rate matches the noise strength closely at low gate counts, this is the most likely culprit.",
                    "bit_flip": f"Bit Flipping (X-error) is cumulative. With {gc_val} gates, the probability of at least one flip built up over time, which explains why the error rate is higher than a single operation's noise.",
                    "phase_flip": "Phase Flipping (Z-error) was detected because you have X-basis (Hadamard) gates enabled. Without those, Z-errors are invisible to standard measurement.",
                    "depolarizing": "Depolarizing noise is the 'worst case' where the qubit state is completely randomized across X, Y, and Z axes, usually caused by complex environmental interaction."
                }
                
                explanation = f"**{pred_label.upper()}** is the primary prediction ({top_p*100:.1f}% confidence).\n\n"
                explanation += f"**Physical Reason**: {physical_map.get(pred_label, '')}\n\n"
                explanation += f"**Key Evidence**: The model's top indicator was **'{top_feat}'**. "
                
                if top_p - next_p < 0.25:
                    comparison = ""
                    if (pred_label == "bit_flip" and next_label == "depolarizing") or (pred_label == "depolarizing" and next_label == "bit_flip"):
                        comparison = "depolarizing noise is a superset that *includes* bit-flips, making them statistically similar at high gate counts."
                    elif (pred_label == "readout" and next_label == "bit_flip") or (pred_label == "bit_flip" and next_label == "readout"):
                        comparison = "low-gate bit-flips look identical to readout flips on a single qubit measurement."
                    elif (next_label == "phase_flip"):
                        comparison = "the presence of basis-shifting gates can sometimes make phase noise look like bit noise if the timing is tight."
                    else:
                        comparison = "there is significant statistical overlap between these noise profiles in small-scale circuits."
                        
                    explanation += f"\n\n⚠️ **Close Second**: **{next_label.upper()}** ({next_p*100:.1f}%). This confusion exists because {comparison}"
                
                st.info(explanation)
                st.markdown("---")
                # 2. Try built-in feature importances (Tree models)
                if hasattr(target_model, 'feature_importances_'):
                    importances = target_model.feature_importances_
                
                # 2. Try MLP weights (sum of absolute weights for each feature)
                elif hasattr(target_model, 'coefs_'):
                    # Mean absolute weight from the first layer
                    importances = np.mean(np.abs(target_model.coefs_[0]), axis=1)
                
                # 3. Fallback: SHAP KernelExplainer (expensive, so we use a small background)
                else:
                    st.write("Using SHAP (this may take a moment)...")
                    # Use a very small background for speed
                    explainer = shap.KernelExplainer(target_model.predict_proba, shap.sample(input_scaled, 1))
                    shap_values = explainer.shap_values(input_scaled)
                    # Extract values for the predicted class
                    importances = np.abs(shap_values[0][pred_class]) if isinstance(shap_values, list) else np.abs(shap_values[0])

                # Normalize importances
                importances = importances / (np.sum(importances) + 1e-6)

                # Plot
                fig, ax = plt.subplots(figsize=(8, 6))
                y_pos = np.arange(len(feature_names))
                # Use a nice color gradient based on importance
                colors = plt.cm.viridis(importances / max(importances))
                
                ax.barh(y_pos, importances, align='center', color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names)
                ax.invert_yaxis()
                ax.set_xlabel('Relative Importance')
                ax.set_title(f'Feature Attribution ({t_name})')
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as xai_e:
                st.warning(f"Could not generate advanced XAI: {xai_e}")
                st.write("Showing basic feature interactions instead.")

elif models is not None:
    st.warning("All models are currently training. Please wait a few seconds and refresh this page.")
else:
    st.error("Essential artifacts not found. Please run `final_train.py` first.")
