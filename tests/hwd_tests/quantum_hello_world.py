from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2 as Sampler,
)

# =========================
# 1. Conexión al servicio
# =========================

service = QiskitRuntimeService(channel="ibm_cloud")

print("Servicio conectado correctamente")

# =========================
# 2. Selección backend
# =========================

backend = service.backend("ibm_basquecountry")

print("Backend seleccionado:", backend.name)

status = backend.status()
print("Operational:", status.operational)
print("Pending jobs:", status.pending_jobs)
print("Status message:", status.status_msg)

# =========================
# 3. Circuito mínimo test
# =========================

qc = QuantumCircuit(1, 1)
qc.measure(0, 0)

print("Circuito creado")

# =========================
# 4. Transpilación hardware-aware
# =========================

tqc = transpile(qc, backend)

print("Circuito transpilado")
print("Depth:", tqc.depth())
print("Ops:", tqc.count_ops())

# =========================
# 5. Sampler runtime
# =========================

sampler = Sampler(mode=backend)

print("Sampler inicializado")

# =========================
# 6. Envío job (shots=4)
# =========================

job = sampler.run([tqc], shots=4)

print("Job enviado correctamente")
print("Job ID:", job.job_id())

# =========================
# 7. Esperar resultado
# =========================

print("Esperando resultado...")

result = job.result()

print("Resultado recibido")

# =========================
# 8. Extraer counts
# =========================

pub_result = result[0]

counts = pub_result.data.c.get_counts()

print("Counts:", counts)

# =========================
# 9. Validación final
# =========================

print("Total shots:", sum(counts.values()))

print("Estado final job:", job.status())