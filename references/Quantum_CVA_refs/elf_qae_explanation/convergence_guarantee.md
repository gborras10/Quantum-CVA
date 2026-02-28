La convergencia al valor verdadero de $\eta = \cos(\theta)$ (y por tanto de la amplitud $a$) en el algoritmo ELF QAE no es accidental, sino que está garantizada por una combinación de principios estadísticos y el diseño activo de los parámetros del circuito.
Aquí tienes los pilares que garantizan esa convergencia:
## 1. **Consistencia del Estimador Bayesiano**
El algoritmo utiliza la Regla de Bayes para actualizar la distribución de probabilidad tras cada medición.

A medida que el número de muestras (disparos cuánticos) aumenta, la distribución posterior de $\theta$ tiende a concentrarse alrededor del valor real, siempre que el modelo de verosimilitud (ELF) sea correcto.
+1


Incluso con ruido, mientras la fidelidad $f$ sea conocida o estimada, el centro de la distribución se desplazará hacia el valor verdadero, aunque la velocidad de esa convergencia sea menor que en un sistema perfecto.


## 2. **Maximización de la Información de Fisher**
La verdadera "garantía" de eficiencia y precisión reside en la Ingeniería de la Función de Verosimilitud. Antes de cada ronda, el ordenador clásico busca los parámetros $\vec{x}$ que maximizan la Información de Fisher.
+1

¿Qué hace esto? La Información de Fisher mide cuánta información contiene una variable aleatoria (el bit $d$ medido) sobre el parámetro desconocido $\theta$.


Al maximizarla, el algoritmo garantiza que la Cota Inferior de Cramér-Rao sea lo más pequeña posible. Esto significa que la varianza de nuestro estimador se reduce al ritmo óptimo permitido por la estadística.


## 3. **El Límite de Heisenberg vs. Ruido (Shot Noise)**
El algoritmo está diseñado para transitar entre dos regímenes de convergencia dependiendo de la calidad del hardware:
Límite de Heisenberg ($O(1/\epsilon)$): En condiciones de bajo ruido, el algoritmo alcanza una precisión que escala linealmente con el número de operaciones, superando la estadística clásica.
+1


Límite de Ruido de Disparo ($O(1/\epsilon^2)$): Si el ruido es muy alto, el algoritmo degrada su comportamiento hasta parecerse a un Monte Carlo clásico, pero sigue convergiendo.


Esta transición suave asegura que, sin importar el nivel de ruido (siempre que se modele en la ELF mediante el factor $f$), el algoritmo llegará al valor real eventualmente.
+2


## 4. **Robustez mediante el factor de fidelidad $f$**
La fórmula de probabilidad incorpora explícitamente la fidelidad del circuito:

$$\mathbb{P}(d|f, \vec{x}) = \frac{1+(-1)^{d}f\langle A|Q^{\dagger}(\vec{x})OQ(\vec{x})|A\rangle}{2} \text{ [cite: 1158]}$$
Si no incluyéramos $f$, el ruido desplazaría la media de nuestra distribución hacia un valor sesgado (bias).
Al incluir $f$, el algoritmo "descuenta" el efecto del ruido, manteniendo la insesgadez del estimador. Básicamente, el algoritmo reconoce que la oscilación de la señal es menor debido al ruido y compensa ese factor en la actualización bayesiana para no "engañarse" con resultados aleatorios.
+3


Resumen de la garantía
La convergencia está garantizada porque el algoritmo diseña activamente sus propias preguntas (los ángulos $\vec{x}$) para que la respuesta (el bit $d$) sea lo más reveladora posible sobre la ubicación exacta de la amplitud real, utilizando un marco bayesiano que es intrínsecamente resistente a errores aleatorios.
+1

¿Te gustaría que viéramos cómo el artículo compara el tiempo de ejecución de esta convergencia frente al Monte Carlo clásico para el caso del CVA?
