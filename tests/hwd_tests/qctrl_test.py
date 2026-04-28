# check_qctrl.py

from __future__ import annotations

import importlib.util
import sys


BACKEND_NAME = "ibm_basquecountry"
CHANNEL = "ibm_quantum_platform"


def check_import(module_name: str) -> bool:
    ok = importlib.util.find_spec(module_name) is not None
    print(f"[{'OK' if ok else 'MISSING'}] {module_name}")
    return ok


def main() -> None:
    print("=" * 80)
    print("Checking local packages")
    print("=" * 80)

    required = [
        "qiskit",
        "qiskit_ibm_runtime",
        "qiskit_ibm_catalog",
    ]

    missing = [m for m in required if not check_import(m)]

    if missing:
        print("\nMissing packages:")
        for m in missing:
            print(f"  - {m}")
        print("\nTry:")
        print("  pip install qiskit qiskit-ibm-runtime qiskit-ibm-catalog")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Checking IBM Quantum access")
    print("=" * 80)

    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        service = QiskitRuntimeService(channel=CHANNEL)
        backend = service.backend(BACKEND_NAME)
        print(f"[OK] IBM Quantum service loaded")
        print(f"[OK] Backend accessible: {backend.name}")
    except Exception as exc:
        print("[ERROR] Could not access IBM Quantum or backend.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print("\nCheck that your IBM account is saved, e.g.:")
        print("  QiskitRuntimeService.save_account(")
        print("      channel='ibm_quantum_platform',")
        print("      token='YOUR_IBM_QUANTUM_TOKEN',")
        print("      overwrite=True,")
        print("  )")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Checking Q-CTRL Performance Management function")
    print("=" * 80)

    from qiskit_ibm_catalog import QiskitFunctionsCatalog

    try:
        catalog = QiskitFunctionsCatalog(channel=CHANNEL)
        perf_mgmt = catalog.load("q-ctrl/performance-management")

        print("[OK] Qiskit Functions Catalog loaded")
        print("[OK] q-ctrl/performance-management is accessible")
        print(f"[INFO] Function object: {perf_mgmt}")

    except Exception as exc:
        print("[ERROR] Q-CTRL Performance Management is not accessible.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print("\nPossible causes:")
        print("  - You do not have access to Qiskit Functions.")
        print("  - Your IBM plan does not include Q-CTRL Performance Management.")
        print("  - qiskit-ibm-catalog is outdated.")
        print("  - Your account/channel configuration is wrong.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Result")
    print("=" * 80)
    print("[SUCCESS] Your environment can access Q-CTRL Performance Management.")
    print("[NOTE] No QPU job has been submitted.")


if __name__ == "__main__":
    main()