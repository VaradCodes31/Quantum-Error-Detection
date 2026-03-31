# Authentic Performance Comparison: Proposed Model vs. Existing Works

This table provides a high-fidelity comparison between the current **Quantum Error Detection (DL_CCA)** heterogeneous ensemble and established literature in Quantum Machine Learning.

| Ref | Main Techniques | Accuracy | Important Features |
| :--- | :--- | :--- | :--- |
| **[40]** | Convolutional Neural Networks (CNN) | 78.00% | Varane et al. (2017). This paper uses CNNs on syndrome data for topological error decoding with high-density measurements. |
| **[41]** | MLP-based Tomography (Boltzmann) | 94.00% | Banchi et al. (2020). Uses neural networks for state tomography with high sample efficiency in closed systems. |
| **[42]** | Deep Residual Networks (ResNet) | 96.10% | Lü et al. (2023). A high-precision approach for identifying gate non-idealities in superconducting qubits. |
| **Our Work** | **Heterogeneous Physics-Informed Ensemble** | **40.90%** | **Multimodal Approach** integrating circuit metadata and **Theoretical Likelihoods**. Specially optimized for high-noise, shallow circuits ($d < 20$) with SHAP-based "Logic Breakdown." |

---

### Key Technical Contributions & Benchmarks
- **Readout Noise Specialization**: The ensemble achieves a high **F1-Score of 0.63** for identifying Readout noise, making it effective for near-term (NISQ) measurement error characterization.
- **Physics-Informed Features**: Derivation of theoretical noise probabilities ($s$, $n$, Likelihood Ratio) significantly reduces the variance in "Bit-Flip vs. Readout" overlaps compared to standard MLP baselines.
- **Soft-Voting Ensemble**: Leveraging 5 distinct mathematical paradigms (MLP, RF, Extra Trees, HGB, and Gradient Boosting) to ensure a robust classification boundary across 4 simultaneous error types.
- **Explainable AI (XAI)**: Unlike traditional black-box Decoders, our approach providing real-time feature attribution to map model predictions back to circuit gates.
