Esta es una explicación rigurosa y detallada del algoritmo Engineered Likelihood Function (ELF) Quantum Amplitude Estimation, basándonos estrictamente en el Apéndice E y la Sección 5 del artículo de Alcázar et al. (2022).
Para entender ELF, debemos desglosarlo en tres componentes fundamentales: el Circuito Parametrizado, el Modelo de Ruido y el Ciclo de Inferencia Bayesiana.

---

# 1. **El Objetivo Matemático**
El objetivo es estimar el valor esperado de un observable. En el contexto de CVA, queremos estimar una amplitud $a$. Formalmente, buscamos el valor esperado $\eta$:

$$\eta = \cos(\theta) = \langle A|O|A\rangle$$
Donde:
- $|A\rangle = \mathcal{A}|0^k\rangle$ es el estado preparado por el circuito ansatz $\mathcal{A}$.


- $O = 2\Pi - I$ es el operador observable, donde $\Pi$ es el proyector sobre el estado que nos interesa (el estado $|1\rangle$ de los qubits ancilla que indican default, payoff, etc.).


- $\theta = \arccos(\eta)$ es el parámetro que estimaremos mediante inferencia.


# 2. **El Hardware: Circuito Parametrizado (Adiós a QPE)**
A diferencia de la QAE convencional que usa la Transformada de Fourier Cuántica (QPE) y un operador de Grover fijo, ELF utiliza una secuencia de operadores unitarios parametrizables.
**A. Los Bloques Constructivos**
El circuito se construye alternando dos tipos de rotaciones "sintonizables" con parámetros reales $x, y \in \mathbb{R}$:

1. Rotación del Oráculo ($U(x)$):
    $$U(x) = e^{ix\Pi}$$
    Este operador aplica una fase que depende del proyector $\Pi$. Es análogo a la parte de "marcado" en Grover, pero con una fase arbitraria $x$ en lugar de $\pi$.
2. Rotación del Estado ($V(y)$):
    $$V(y) = \mathcal{A} e^{iy |0\rangle\langle 0|} \mathcal{A}^\dagger$$
    Este operador es una rotación alrededor del estado inicial $|0^k\rangle$, "intercalada" por el operador de preparación del estado $\mathcal{A}$.

**B. El Circuito Completo (La Profundidad es Variable)**
El circuito ELF de profundidad $L$ aplica una secuencia de estos operadores:
$$Q(\vec{x})|A\rangle = V(x_{2L})U(x_{2L-1})\dots V(x_2)U(x_1)|A\rangle$$

Aquí, $\vec{x} = (x_1, x_2, \dots, x_{2L})$ es un vector de parámetros clásicos que podemos elegir ("diseñar") libremente antes de ejecutar el circuito.
# 3. **La Función de Verosimilitud (Likelihood Function)**
Una vez ejecutado el circuito $Q(\vec{x})$, realizamos una medición proyectiva $\{\Pi, I-\Pi\}$. Obtenemos un resultado binario: $d=1$ (éxito) o $d=0$ (fracaso).

### **A. Caso Ideal**
La probabilidad de medir $d$ dado un conjunto de parámetros $\vec{x}$ es:


$$\mathbb{P}(d|\vec{x}) = \frac{1 + (-1)^d \langle A | Q^\dagger(\vec{x}) O Q(\vec{x}) | A \rangle}{2}$$
Esta fórmula conecta directamente nuestro resultado de medición con el valor esperado del operador $O$ transformado por nuestro circuito parametrizado.
### **B. Incorporación del Ruido (Ingeniería Robusta)**
Aquí reside la gran ventaja de ELF para CVA. Asumimos que el circuito tiene una fidelidad $f \in [0, 1]$. El modelo incorpora este factor explícitamente en la función de verosimilitud:
+1


$$\mathbb{P}(d|f, \vec{x}) = \frac{1 + (-1)^d f \langle A | Q^\dagger(\vec{x}) O Q(\vec{x}) | A \rangle}{2}$$
- **Interpretación**: Si el hardware es muy ruidoso ($f$ bajo), el término de la derecha se hace pequeño y la probabilidad $\mathbb{P}(d)$ se acerca a 0.5 (ruido aleatorio). El algoritmo "sabe" esto y ajustará su confianza en la estimación acorde a la calidad del hardware.
# 4. **El Algoritmo: Ciclo de Inferencia Bayesiana**
El proceso no es de "una sola pasada" (one-shot) como en QPE, sino iterativo.
- **Paso 0: Inicialización (Prior) Comenzamos con una distribución** de probabilidad inicial para $\eta$ (o $\theta$), típicamente una Gaussiana, que representa nuestra ignorancia inicial.

- **Paso 1: Ingeniería de la Función (Maximización de Información)**
    Antes de tocar el ordenador cuántico, el ordenador clásico calcula qué parámetros $\vec{x}$ usar.
    El objetivo es maximizar la ganancia de información (Information Gain) de la siguiente medición.
    Esto se logra maximizando la Información de Fisher de la ELF en la ronda actual.


    Intuición: Si nuestra distribución actual dice que el valor está cerca de $0.5$, diseñamos el circuito (elegimos $\vec{x}$) tal que la función de verosimilitud tenga una pendiente muy pronunciada alrededor de $0.5$. Así, si el valor real se desvía mínimamente, la probabilidad de medir 0 o 1 cambiará drásticamente, dándonos mucha información.
    - **Paso 2: Ejecución Cuántica** 
    Ejecutamos el circuito $Q(\vec{x})$ en el procesador cuántico con los parámetros optimizados y medimos el bit $d$.

- **Paso 3: Actualización Bayesiana** 
    Usamos el Teorema de Bayes para actualizar nuestra distribución de conocimiento sobre $\theta$:


    $$P_{posterior}(\theta | d) \propto \mathbb{P}(d|f, \vec{x}, \theta) \cdot P_{prior}(\theta)$$
    Dado que en la "región crítica" la ELF se asemeja a una función sinusoidal, esta actualización se puede realizar eficientemente sin integración numérica costosa.

- **Paso 4: Convergencia** 
    Repetimos los pasos 1-3. Con cada iteración, la distribución Gaussiana se vuelve más estrecha (menor varianza). El algoritmo termina cuando la varianza es suficientemente pequeña. La media de la distribución final de $\theta$ se convierte en nuestra estimación de $\eta = \cos(\theta)$, y por tanto del valor CVA.