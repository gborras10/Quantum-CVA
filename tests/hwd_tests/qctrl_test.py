from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit_ibm_catalog import QiskitFunctionsCatalog
from qiskit_ibm_runtime import QiskitRuntimeService


def build_tiny_qpu_circuit() -> QuantumCircuit:
    """
    Circuito mínimo para comprobar ejecución real:
    |0> --H-- measure

    No lo transpiles: Q-CTRL Performance Management espera circuitos abstractos.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def compact(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="ibm_quantum_platform")
    parser.add_argument("--backend", default="ibm_basquecountry")
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--run-qpu", action="store_true")
    parser.add_argument("--save-job", default="qctrl_pm_test_job.json")
    args = parser.parse_args()

    print("=" * 90)
    print("1) Comprobando IBM Quantum Runtime")
    print("=" * 90)

    service = QiskitRuntimeService(channel=args.channel)
    backends = service.backends()
    backend_names = sorted(b.name for b in backends)

    print(f"Channel: {args.channel}")
    print(f"Backends disponibles: {backend_names}")

    if args.backend not in backend_names:
        raise RuntimeError(
            f"No veo el backend {args.backend!r}. "
            f"Backends disponibles: {backend_names}"
        )

    backend = service.backend(args.backend)
    status = backend.status()
    print(f"Backend objetivo: {backend.name}")
    print(f"Operational: {getattr(status, 'operational', None)}")
    print(f"Pending jobs: {getattr(status, 'pending_jobs', None)}")
    print(f"Status message: {getattr(status, 'status_msg', None)}")

    print("\n" + "=" * 90)
    print("2) Comprobando acceso a Q-CTRL Performance Management")
    print("=" * 90)

    catalog = QiskitFunctionsCatalog(channel=args.channel)

    try:
        perf_mgmt = catalog.load("q-ctrl/performance-management")
        print("\n" + "=" * 90)
        print("2b) Inspeccionando atributos de la función Q-CTRL")
        print("=" * 90)

        for name in dir(perf_mgmt):
            if "backend" in name.lower() or "device" in name.lower() or "support" in name.lower():
                print(name)

        try:
            print("\nMetadata:")
            print(perf_mgmt.metadata)
        except Exception as exc:
            print("No metadata:", repr(exc))

        try:
            print("\nDetails:")
            print(perf_mgmt.details())
        except Exception as exc:
            print("No details():", repr(exc))
    except Exception as exc:
        print("\nERROR cargando q-ctrl/performance-management.")
        print("Causas probables:")
        print("  - No tienes licencia/entitlement de Q-CTRL en tu instancia.")
        print("  - Tu cuenta no está en Premium/Flex/On-Prem o no está habilitada.")
        print("  - Falta qiskit-ibm-catalog o estás autenticado en otro channel/instance.")
        raise exc

    print("OK: función cargada correctamente.")
    print(f"Function object: {perf_mgmt}")

    if not args.run_qpu:
        print("\nNo se ha enviado nada a QPU.")
        print("Para lanzar la prueba mínima real:")
        print(
            f"python {Path(__file__).name} "
            f"--channel {args.channel} "
            f"--backend {args.backend} "
            f"--shots {args.shots} "
            f"--run-qpu"
        )
        return

    print("\n" + "=" * 90)
    print("3) Enviando prueba mínima a QPU vía Q-CTRL Performance Management")
    print("=" * 90)

    qc = build_tiny_qpu_circuit()
    pubs = [(qc,)]

    print("Circuito abstracto enviado:")
    print(qc.draw(output="text"))
    print(f"Shots solicitados: {args.shots}")

    t0 = time.perf_counter()

    job = perf_mgmt.run(
        primitive="sampler",
        pubs=pubs,
        backend_name=args.backend,
        options={
            "default_shots": args.shots,
            "job_tags": ["qctrl-pm-connectivity-test", "tiny-1q-test"],
        },
    )

    def get_job_id(job) -> str | None:
        jid = getattr(job, "job_id", None)
        if callable(jid):
            return jid()
        return jid


    job_info = {
        "backend": args.backend,
        "channel": args.channel,
        "shots": args.shots,
        "job_id": get_job_id(job),
        "status_initial": str(job.status()),
    }

    Path(args.save_job).write_text(json.dumps(job_info, indent=2), encoding="utf-8")

    print(f"Job enviado. Info guardada en: {args.save_job}")
    print(compact(job_info))

    print("\nEsperando resultado...")
    result = job.result()

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 90)
    print("4) Resultado")
    print("=" * 90)

    pub_result = result[0]

    try:
        counts = pub_result.data.c.get_counts()
    except Exception:
        counts = None

    print(f"Status final: {job.status()}")
    print(f"Elapsed wall-clock local: {elapsed:.3f} s")

    if counts is not None:
        print(f"Counts: {counts}")
    else:
        print("No he podido extraer counts con pub_result.data.c.get_counts().")
        print("Resultado bruto:")
        print(result)

    print("\nMetadata:")
    print(compact(getattr(result, "metadata", None)))
    print("\nPubResult data:")
    print(compact(getattr(pub_result, "data", None)))


if __name__ == "__main__":
    main()