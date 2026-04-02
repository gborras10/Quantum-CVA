import json
import datetime
import pathlib
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel

class QuantumHardwareArchitect:
    """
    Class to manage quantum hardware profiles, including fetching real-time data 
    from IBM Cloud and persisting it locally. Provides methods to analyze hardware 
    characteristics relevant for quantum algorithm design.
    """

    def __init__(self, backend_name: str = None, file_path: str = None):
        self.backend_name = backend_name
        self.data = None
        self.noise_model = None
        
        if file_path:
            self.load_from_json(file_path)
        elif backend_name:
            self.fetch_remote_data()

    def fetch_remote_data(self):
        """Extracts hardware information from the specified IBM Quantum backend."""
        print(f"[STATUS] Starting data retrieval for backend: {self.backend_name}...")
        try:
            service = QiskitRuntimeService(channel="ibm_cloud")
            backend = service.backend(self.backend_name)
            
            print(f"[STATUS] Downloading data for {self.backend_name}...")
            config = backend.configuration()
            props = backend.properties()
            noise_model = NoiseModel.from_backend(backend)
            
            self.noise_model = noise_model
            self.data = {
                "metadata": {
                    "backend_name": self.backend_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "n_qubits": config.n_qubits,
                    "basis_gates": config.basis_gates,
                    "processor_type": getattr(config, 'processor_type', 'unknown')
                },
                "configuration": config.to_dict(),
                "properties": props.to_dict(),
                "noise_model": noise_model.to_dict()
            }
            print("[INFO] Hardware data retrieved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to retrieve backend information: {str(e)}")

    def save_to_json(self, directory: str = "./data/profiles"):
        """Persists the current profile in a structured JSON file."""
        if not self.data:
            print("[ERROR] No data available to save.")
            return

        path = pathlib.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = path / f"profile_{self.backend_name}_{timestamp}.json"
        
        with open(file_name, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"[INFO] Profile saved to {file_name}")
        return str(file_name)

    def load_from_json(self, file_path: str):
        """Loads a previous profile from the file system."""
        print(f"[STATUS] Loading profile from {file_path}...")
        with open(file_path, "r") as f:
            self.data = json.load(f)
        
        self.backend_name = self.data["metadata"]["backend_name"]
        self.noise_model = NoiseModel.from_dict(self.data["noise_model"])
        print(f"[INFO] {self.backend_name} profile loaded successfully.")

    def get_coherence_metrics(self):
        """Returns descriptive statistics of the T1 and T2 times."""
        props = self.data["properties"]
        # Extraction of T1 and T2 converting from seconds to microseconds
        t1_list = [p[0]['value'] * 1e6 for p in props['qubits'] if p[0]['name'] == 'T1']
        t2_list = [p[1]['value'] * 1e6 for p in props['qubits'] if p[1]['name'] == 'T2']
        
        return {
            "t1_median": np.median(t1_list),
            "t1_std": np.std(t1_list),
            "t2_median": np.median(t2_list),
            "t2_std": np.std(t2_list)
        }

    def get_gate_error_metrics(self):
        """Analyzes the error rates of single-qubit and two-qubit gates."""
        props = self.data["properties"]
        gate_errors_1q = []
        gate_errors_2q = []
        
        for gate in props['gates']:
            error = gate['parameters'][0]['value']
            if len(gate['qubits']) == 1:
                gate_errors_1q.append(error)
            else:
                gate_errors_2q.append(error)
                
        return {
            "median_1q_error": np.median(gate_errors_1q),
            "median_2q_error": np.median(gate_errors_2q),
            "max_2q_error": np.max(gate_errors_2q)
        }

    def get_connectivity_map(self):
        """Returns the adjacency list (Coupling Map) of the chip."""
        return self.data["configuration"]["coupling_map"]