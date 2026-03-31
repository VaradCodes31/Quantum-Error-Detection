import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from src.preprocessing import load_data, preprocess_data
from src.deep_learning import train_deep_learning_model, train_mlp_model
from src.attention_model import train_attention_model
from src.cnn_model import train_cnn_model
from src.transformer_model import train_transformer_model
from src.ensemble import get_ensemble_model
from src.visualization import plot_training_history
from src.xai_utils import explain_with_shap, get_integrated_gradients, plot_feature_importance

def main():
    path = "data/quantum_multiclass_dataset.csv"

    # Load and Preprocess
    df = load_data(path)
    # Using 2000 samples for faster model generation
    df = df.sample(n=min(len(df), 2000), random_state=42)
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df)
    joblib.dump(feature_names, 'results/feature_names.pkl')

    # 1️⃣ MLP Model
    print("\nTraining MLP Model (Fast Version)...")
    mlp_acc, mlp_model, _ = train_mlp_model(X_train, X_test, y_train, y_test)

    # 2️⃣ ResNet Model
    print("\nTraining ResNet Model (Fast Version)...")
    resnet_params = {'dense_units': 128, 'dropout_rate': 0.1, 'epochs': 20, 'batch_size': 128}
    resnet_acc, resnet_model, history = train_deep_learning_model(X_train, X_test, y_train, y_test, params=resnet_params)

    # 3️⃣ Gated Attention Model
    print("\nTraining Gated Attention Model (Fast Version)...")
    att_params = {'dense_units': 64, 'dropout_rate': 0.1, 'epochs': 20, 'batch_size': 128}
    att_acc, att_model = train_attention_model(X_train, X_test, y_train, y_test, params=att_params)

    # 4️⃣ 1D-CNN Model
    print("\nTraining 1D-CNN Model (Fast Version)...")
    cnn_acc, cnn_model, _ = train_cnn_model(X_train, X_test, y_train, y_test)

    # 5️⃣ Transformer Model
    print("\nTraining Transformer Model (Fast Version)...")
    trans_acc, trans_model, _ = train_transformer_model(X_train, X_test, y_train, y_test)

    # 6️⃣ Ensemble Model (Averaging all 5 DL models)
    print("\nEvaluating Multi-Architecture Ensemble...")
    models = [mlp_model, resnet_model, att_model, cnn_model, trans_model]
    ensemble = get_ensemble_model(models) # Equal weights by default
    ens_acc = ensemble.evaluate(X_test, y_test)
    
    # 7️⃣ Visualizations and Reports
    plot_training_history(history)

    print("\n" + "="*35)
    print("FINAL PURE-DL MODEL COMPARISON")
    print("="*35)
    print(f"MLP Accuracy:          {mlp_acc:.4f}")
    print(f"ResNet Accuracy:       {resnet_acc:.4f}")
    print(f"Attention Accuracy:    {att_acc:.4f}")
    print(f"1D-CNN Accuracy:       {cnn_acc:.4f}")
    print(f"Transformer Accuracy:  {trans_acc:.4f}")
    print(f"DL ENSEMBLE Accuracy:  {ens_acc:.4f}")
    print("="*35)

    y_pred = ensemble.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    print("\nClassification Report (DL Ensemble):\n", report)

    # Log results
    with open("results_log_pure_dl.txt", "w") as f:
        f.write("Quantum Error Detection - Pure Deep Learning Results\n")
        f.write("="*50 + "\n")
        f.write(f"MLP:          {mlp_acc:.4f}\n")
        f.write(f"ResNet:       {resnet_acc:.4f}\n")
        f.write(f"Attention:    {att_acc:.4f}\n")
        f.write(f"1D-CNN:       {cnn_acc:.4f}\n")
        f.write(f"Transformer:  {trans_acc:.4f}\n")
        f.write(f"DL Ensemble:  {ens_acc:.4f}\n")
        f.write("\n" + report)


if __name__ == "__main__":
    main()