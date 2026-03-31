from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import os

def train_optimized_ensemble(X_train, X_test, y_train, y_test):
    """
    Final, most accurate 5-model soft-voting ensemble.
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    print("🚀 Training Final 5-Model Ensemble...")
    
    clf1 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=500, max_depth=25, class_weight='balanced', random_state=42, n_jobs=-1)
    clf3 = ExtraTreesClassifier(n_estimators=500, max_depth=25, class_weight='balanced', random_state=42, n_jobs=-1)
    clf4 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
    clf5 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ('mlp', clf1), 
            ('rf', clf2), 
            ('et', clf3), 
            ('hgb', clf4), 
            ('gb', clf5)
        ],
        voting='soft',
        weights=[1, 4, 3, 1, 1] 
    )

    import time
    start = time.time()
    print("  > Fitting Ensemble...", end=" ", flush=True)
    ensemble.fit(X_train, y_train)
    
    # 3. Individual Benchmarking
    benchmarks = {}
    print("\n📊 Model Benchmarking:")
    for name, clf in ensemble.named_estimators_.items():
        score = clf.score(X_test, y_test)
        benchmarks[name.upper()] = score
        print(f"  - {name.upper()}: {score:.4f}")
    
    # Add Ensemble score
    ensemble_score = ensemble.score(X_test, y_test)
    benchmarks['ENSEMBLE'] = ensemble_score
    print(f"  - ENSEMBLE: {ensemble_score:.4f}")

    joblib.dump(ensemble, 'results/best_ensemble_model.joblib')
    joblib.dump(benchmarks, 'results/benchmarks.pkl')
    
    print(f"Done ({time.time()-start:.2f}s)")

    return ensemble