# Algoritmos Agénticos

Repositorio dedicado al estudio, desarrollo y experimentación con **algoritmos agénticos basados en modelos de lenguaje (LLM)**. El objetivo es explorar diferentes arquitecturas de agentes, sus estrategias de razonamiento, planificación y toma de decisiones, así como sus aplicaciones prácticas.

## Algoritmos implementados

### 1. ReAct (Reasoning and Acting)

ReAct combina el razonamiento del modelo con la ejecución de acciones. El agente alterna entre analizar la situación, seleccionar herramientas, observar sus resultados y decidir el siguiente paso.

**Características principales:**
- Alternancia entre razonamiento y acción.
- Uso de herramientas externas para obtener información o ejecutar tareas.
- Incorporación de las observaciones en las decisiones posteriores.
- Adaptación del comportamiento según los resultados obtenidos.

### 2. Plan-and-Execute

Plan-and-Execute separa la planificación de una tarea de su ejecución. Un planificador descompone el problema en pasos y un ejecutor realiza las acciones necesarias para completar el plan.

**Características principales:**
- Descomposición de problemas complejos en tareas más pequeñas.
- Generación de un plan inicial.
- Ejecución secuencial de los pasos planificados.
- Posibilidad de revisar o actualizar el plan según los resultados.

### 3. Tree of Thoughts (ToT)

Tree of Thoughts explora múltiples alternativas de razonamiento en forma de árbol. En lugar de seguir una única línea de pensamiento, el agente genera distintas opciones, las evalúa y prioriza aquellas con mayor potencial para resolver el problema.

**Características principales:**
- Generación de varias alternativas de solución.
- Evaluación de los pensamientos mediante criterios definidos.
- Selección y expansión de las alternativas más prometedoras.
- Descarte de caminos inválidos o poco prometedores.


## Tecnologías

- **Python** — implementación de los algoritmos.
- **LangChain** — integración con modelos de lenguaje y herramientas.
- **LangGraph** — construcción de flujos de trabajo y grafos de agentes.
- **FastAPI** — exposición de funcionalidades mediante una API REST.
- **Pandas** — procesamiento y análisis de datos.
- **Groq** — acceso a modelos de lenguaje, cuando corresponda.


## Instalación y configuración

Clona el repositorio e instala las dependencias:

```bash
git clone <URL_DEL_REPOSITORIO>
cd algoritmos-agenticos
pip install -r requirements.txt
```

Configura las variables de entorno necesarias en un archivo `.env`. Por ejemplo, si utilizas Groq:

```env
GROQ_API_KEY=tu_api_key
```

No incluyas claves API ni credenciales reales en el repositorio.



## Trabajo futuro

- Incorporar nuevas arquitecturas agénticas.
- Comparar las estrategias de razonamiento y planificación.
- Medir latencia, número de llamadas al LLM y costes.
- Evaluar la calidad de las soluciones generadas.
- Explorar mecanismos de reflexión, búsqueda y mejora iterativa.

## Licencia

La licencia del proyecto se definirá próximamente.
