import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error


# ------------------------------
# Generate different noise models
# ------------------------------
def create_noise_model(error_type, strength=0.05):

    noise_model = NoiseModel()
    # Instructions that are GATES only (no measure)
    gate_instructions = ["id", "rz", "sx", "x", "h", "z"]

    # ---- Readout error (Measurement only) ----
    if error_type == "readout":
        readout = ReadoutError([
            [1 - strength, strength],
            [strength, 1 - strength]
        ])
        noise_model.add_readout_error(readout, [0])

    # ---- Bit flip (Gates only) ----
    elif error_type == "bit_flip":
        # Concentrate Bit Flip on GATES
        bitflip = pauli_error([("X", strength), ("I", 1 - strength)])
        noise_model.add_all_qubit_quantum_error(bitflip, gate_instructions)

    # ---- Phase flip (Gates only) ----
    elif error_type == "phase_flip":
        # Phase flip is Z-axis only
        phaseflip = pauli_error([("Z", strength), ("I", 1 - strength)])
        noise_model.add_all_qubit_quantum_error(phaseflip, gate_instructions)

    # ---- Depolarizing (Gates only) ----
    elif error_type == "depolarizing":
        # Depolarizing is generalized: X, Y, Z
        dep = depolarizing_error(strength, 1)
        noise_model.add_all_qubit_quantum_error(dep, gate_instructions)

    return noise_model


# ------------------------------
# Random circuit generator
# ------------------------------
def generate_random_circuit():

    qc = QuantumCircuit(1, 1)

    # Randomly prepare 0 or 1
    prepared_state = np.random.choice([0, 1])

    if prepared_state == 1:
        qc.x(0)

    # Add random gates (increase count for more complexity)
    gate_count = np.random.randint(5, 20)

    for _ in range(gate_count):
        gate = np.random.choice(["h", "x", "z", "rz", "sx"])

        if gate == "h": qc.h(0)
        elif gate == "x": qc.x(0)
        elif gate == "z": qc.z(0)
        elif gate == "rz": qc.rz(np.pi/4, 0)
        elif gate == "sx": qc.sx(0)

    qc.measure(0, 0)

    return qc, prepared_state, gate_count


# ------------------------------
# Main dataset generator
# ------------------------------
def generate_quantum_dataset(
    runs=500,
    shots=1024,
    output_file="quantum_multiclass_dataset.csv"
):

    error_types = ["readout", "bit_flip", "phase_flip", "depolarizing"]
    records = []

    for _ in range(runs):

        # ---- Select error type ----
        error_type = np.random.choice(error_types)
        strength = np.random.uniform(0.01, 0.45)

        noise_model = create_noise_model(error_type, strength)

        backend = AerSimulator(noise_model=noise_model)

        # ---- Random circuit ----
        qc, prepared_state, gate_count = generate_random_circuit()

        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()

        count_0 = counts.get("0", 0)
        count_1 = counts.get("1", 0)

        # ---- Compute Ideal Result ----
        ideal_backend = AerSimulator()
        ideal_result = ideal_backend.run(qc, shots=shots).result()
        ideal_count_0 = ideal_result.get_counts().get("0", 0)

        # ---- Compute error ----
        error_rate = abs(ideal_count_0 - count_0) / shots

        records.append({
            "prepared_state": prepared_state,
            "measured_0": count_0,
            "measured_1": count_1,
            "error_rate": error_rate,
            "error_type": error_type,
            "noise_strength": strength,
            "gate_count": gate_count,
            "circuit_depth": qc.depth(),
            "has_h_gate": 1 if 'h' in [gate.name for gate, _, _ in qc.data] else 0
        })

    dataset = pd.DataFrame(records)
    dataset.to_csv(output_file, index=False)

    print(f"Dataset generated: {len(dataset)} rows")
    print(dataset.head())


if __name__ == "__main__":
    generate_quantum_dataset(runs=20000)