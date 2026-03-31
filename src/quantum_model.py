from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


def train_quantum_model(X_train, X_test, y_train, y_test):

    print("\nTraining Quantum QSVC...")

    subset = 400

    X_train_small = X_train[:subset]
    y_train_small = y_train[:subset]

    X_test_small = X_test[:200]
    y_test_small = y_test[:200]

    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)

    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    qsvc = QSVC(quantum_kernel=quantum_kernel)

    qsvc.fit(X_train_small, y_train_small)

    score = qsvc.score(X_test_small, y_test_small)

    print("Quantum QSVC Accuracy:", score)

    return score