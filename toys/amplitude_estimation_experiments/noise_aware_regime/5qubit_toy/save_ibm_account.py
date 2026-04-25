from qiskit_ibm_runtime import QiskitRuntimeService

print("Iniciando prueba de conexión...")

try:
    # Intenta conectar usando la cuenta guardada de IBM Cloud
    service = QiskitRuntimeService(channel="ibm_cloud")
    print("✅ ¡Éxito! Conectado correctamente a IBM Cloud.")
    print("Instancia activa:", service.active_account()['instance'])
    
    # Comprobar si podemos ver los backends
    backends = service.backends()
    print(f"Tienes acceso a {len(backends)} backends cuánticos.")
    
except Exception as e:
    print("❌ Error de conexión:", e)