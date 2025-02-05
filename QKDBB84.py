from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import random
import numpy as np
import os
from datetime import datetime
import time


class BB84Protocol:
    def __init__(self, use_real_device=False, backend_name=None):
        self.use_real_device = use_real_device
        self.backend_name = backend_name
        self.service = None
        self.sampler = None
        self.backend_supports_dynamic_circuits = False
        if use_real_device:
            try:
                self.service = QiskitRuntimeService()
                if backend_name:
                    self.backend = self.service.backend(backend_name)
                else:
                    available_backends = self.service.backends(
                        filters=lambda x: x.num_qubits >= 5 and not x.simulator
                    )
                    self.backend = min(available_backends,
                                       key=lambda x: x.status().pending_jobs)
                print(f"Using quantum backend: {self.backend.name}")
                # Check if the backend supports dynamic circuits
                self.backend_supports_dynamic_circuits = self.backend.configuration().dynamic_circuits
                print(f"Backend supports dynamic circuits: {self.backend_supports_dynamic_circuits}")
                self.sampler = Sampler(mode=self.backend)
                self.sampler.options.default_shots = 1024
            except Exception as e:
                print(f"Error connecting to IBM Quantum: {e}")
                print("Falling back to simulator")
                self.use_real_device = False
        if not self.use_real_device:
            self.backend = AerSimulator()  # Use AerSimulator for local simulation
            self.backend_supports_dynamic_circuits = True  # Simulators support dynamic circuits
            self.sampler = Sampler(mode=self.backend)
            self.sampler.options.default_shots = 1024

    def get_random_inputs(self, n):
        alice_bits = [random.randint(0, 1) for _ in range(n)]
        alice_bases = [random.randint(0, 1) for _ in range(n)]
        bob_bases = [random.randint(0, 1) for _ in range(n)]
        return alice_bits, alice_bases, bob_bases

    def create_quantum_circuit(self, bits, bases, measurement_bases, include_mid_circuit_measurement=False):
        """Create quantum circuit for encoding and measurement"""
        n = len(bits)
        circuit = QuantumCircuit(n, n)

        # Encode the bits into qubits
        for i in range(n):
            if bases[i] == 0:  # Z-basis
                if bits[i] == 1:
                    circuit.x(i)
            else:  # X-basis
                if bits[i] == 0:
                    circuit.h(i)
                else:
                    circuit.x(i)
                    circuit.h(i)

        # Mid-circuit measurement (if enabled)
        if include_mid_circuit_measurement:
            for i in range(n):
                circuit.measure(i, i)
                circuit.reset(i)

        # Final measurement in Bob's bases
        for i in range(n):
            if measurement_bases[i] == 1:  # X-basis measurement
                circuit.h(i)
            circuit.measure(i, i)
        return circuit

    def simulate_or_run_circuit(self, circuit, shots=1024):
        try:
            print("Submitting job...")
            transpiled_circuit = transpile(circuit, self.backend)
            job = self.sampler.run([transpiled_circuit])
            print("Waiting for results...")
            result = job.result()[0]

            if hasattr(result.data, "meas"):
                counts = result.data.meas.get_counts()
            elif hasattr(result.data, "c"):
                bitarray = result.data.c
                counts = bitarray.get_counts()
            else:
                raise ValueError("Unsupported result structure.")

            n_qubits = circuit.num_qubits
            measurements = np.zeros(n_qubits)
            total_shots = sum(counts.values())
            for bitstring, count in counts.items():
                bits = [int(x) for x in bitstring[::-1]]
                measurements += np.array(bits) * (count / total_shots)
            return [int(round(x)) for x in measurements]
        except Exception as e:
            print(f"Error running the circuit: {e}")
            return None

    def extract_shared_key(self, alice_bases, bob_bases, bob_results, alice_bits):
        shared_key = []
        shared_indices = []
        for i in range(len(alice_bases)):
            if alice_bases[i] == bob_bases[i]:
                shared_key.append(bob_results[i])
                shared_indices.append(i)
        return shared_key, shared_indices

    def verify_shared_key(self, alice_bits, bob_results, shared_indices):
        total_bits = len(shared_indices)
        matches = 0
        for i in shared_indices:
            if alice_bits[i] == bob_results[i]:
                matches += 1
        success_rate = (matches / total_bits * 100) if total_bits > 0 else 0
        return success_rate


def main():
    print("BB84 Quantum Key Distribution Protocol")
    print("=====================================")

    # Prompt for number of runs
    num_runs = int(input("How many times would you like to run the protocol? "))

    # Create the 'outputs' directory
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    # Generate a timestamped directory name using local time (with seconds)
    timestamp = datetime.now().strftime("%m%d%y_%H%M%S")  # Time stamped directories
    run_dir = os.path.join(outputs_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Open a log file for storing console output
    log_file_path = os.path.join(run_dir, "data_log.txt")
    with open(log_file_path, "w") as log_file:
        def log_and_print(message):
            print(message)
            log_file.write(message + "\n")

        for run in range(num_runs):
            log_and_print(f"\n=== Run {run + 1} of {num_runs} ===")

            # Start timing the run
            start_time = time.time()

            # Get execution mode
            use_real_device = input("Use real quantum device? (yes/no): ").lower().strip() == 'yes'
            eve_present = input("Enable Eve (eavesdropper)? (yes/no): ").lower().strip() == 'yes'

            backend_name = None
            if use_real_device:
                log_and_print("\nConnecting to IBM Quantum...")
                try:
                    service = QiskitRuntimeService()
                    backends = service.backends(
                        filters=lambda x: x.num_qubits >= 5 and not x.simulator
                    )
                    for idx, backend in enumerate(backends):
                        status = backend.status()
                        log_and_print(f"{idx + 1}. {backend.name} - "
                                      f"{status.pending_jobs} jobs in queue")
                    choice = input("\nSelect backend number "
                                   "(press Enter for automatic selection): ").strip()
                    if choice.isdigit() and 0 < int(choice) <= len(backends):
                        backend_name = backends[int(choice) - 1].name
                except Exception as e:
                    log_and_print(f"Error accessing IBM Quantum: {e}")
                    log_and_print("Falling back to simulator")
                    use_real_device = False

            bb84 = BB84Protocol(use_real_device=use_real_device, backend_name=backend_name)

            n_bits = int(input("\nEnter number of bits to generate (5-20 recommended): "))
            alice_bits, alice_bases, bob_bases = bb84.get_random_inputs(n_bits)
            log_and_print("\nInitial values:")
            log_and_print(f"Alice's bits: {alice_bits}")
            log_and_print(f"Alice's bases: {alice_bases}")
            log_and_print(f"Bob's bases: {bob_bases}")

            if eve_present:
                if bb84.backend_supports_dynamic_circuits:
                    log_and_print("\nBackend supports dynamic circuits. Using mid-circuit measurements for Eve.")
                    # Create a single circuit with Eve's interception
                    circuit = bb84.create_quantum_circuit(alice_bits, alice_bases, bob_bases,
                                                          include_mid_circuit_measurement=True)
                    sim_or_real = "REAL" if use_real_device else "SIM"
                    circuit_image_path = os.path.join(run_dir, f"{sim_or_real}_RUN{run + 1}.png")
                    circuit_drawer(circuit, output='mpl').savefig(circuit_image_path)
                    log_and_print(f"Circuit diagram saved to: {circuit_image_path}")

                    # Run the circuit
                    bob_results = bb84.simulate_or_run_circuit(circuit)
                    if bob_results is None:
                        log_and_print("Failed to execute the circuit.")
                        continue
                else:
                    log_and_print(
                        "\nBackend does not support dynamic circuits. Falling back to split-circuit approach.")
                    # Step 1: Eve's measurement
                    eve_bases = [random.randint(0, 1) for _ in range(n_bits)]
                    eve_circuit = bb84.create_quantum_circuit(alice_bits, alice_bases, eve_bases,
                                                              include_mid_circuit_measurement=False)
                    eve_results = bb84.simulate_or_run_circuit(eve_circuit)
                    if eve_results is None:
                        log_and_print("Failed to execute Eve's circuit.")
                        continue
                    log_and_print(f"Eve's measurement results: {eve_results}")

                    # Step 2: Bob's measurement
                    bob_circuit = bb84.create_quantum_circuit(eve_results, alice_bases, bob_bases,
                                                              include_mid_circuit_measurement=False)
                    sim_or_real = "REAL" if use_real_device else "SIM"
                    circuit_image_path = os.path.join(run_dir, f"{sim_or_real}_RUN{run + 1}.png")
                    circuit_drawer(bob_circuit, output='mpl').savefig(circuit_image_path)
                    log_and_print(f"Circuit diagram saved to: {circuit_image_path}")

                    # Run Bob's circuit
                    bob_results = bb84.simulate_or_run_circuit(bob_circuit)
                    if bob_results is None:
                        log_and_print("Failed to execute Bob's circuit.")
                        continue
            else:
                # No Eve; create a single circuit for Bob
                circuit = bb84.create_quantum_circuit(alice_bits, alice_bases, bob_bases,
                                                      include_mid_circuit_measurement=False)
                sim_or_real = "REAL" if use_real_device else "SIM"
                circuit_image_path = os.path.join(run_dir, f"{sim_or_real}_RUN{run + 1}.png")
                circuit_drawer(circuit, output='mpl').savefig(circuit_image_path)
                log_and_print(f"Circuit diagram saved to: {circuit_image_path}")

                # Run the circuit
                bob_results = bb84.simulate_or_run_circuit(circuit)
                if bob_results is None:
                    log_and_print("Failed to execute the circuit.")
                    continue

            log_and_print(f"\nBob's measurement results: {bob_results}")

            # Extract and verify shared key
            shared_key, shared_indices = bb84.extract_shared_key(
                alice_bases, bob_bases, bob_results, alice_bits
            )
            log_and_print(f"\nShared key: {shared_key}")
            success_rate = bb84.verify_shared_key(alice_bits, bob_results, shared_indices)
            log_and_print(f"\nSuccess rate: {success_rate:.2f}%")

            # Detect Eve's presence
            if success_rate < 90:  # Arbitrary threshold for detecting eavesdropping
                log_and_print("\nWarning: High error rate detected! Possible eavesdropping.")
            else:
                log_and_print("\nSuccess: Valid shared key established!")

            # End timing the run
            end_time = time.time()
            runtime = end_time - start_time
            log_and_print(f"\nRun {run + 1} completed in {runtime:.2f} seconds.\n")


if __name__ == "__main__":
    main()
