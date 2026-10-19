"""
===========================================================
¿CUÁNDO IMPLEMENTAR UN AGENTE PLAN-AND-EXECUTE?
===========================================================

Plan-and-Execute separa el proceso en dos fases principales:

    1. PLANNER
       El LLM analiza la petición y genera un plan completo.

    2. EXECUTOR
       Ejecuta los pasos del plan en el orden establecido.

Finalmente:

    3. FINAL
       Otro paso LLM resume los resultados y responde al usuario.


ESQUEMA:

                    ┌──────────────┐
                    │   PLANNER    │
                    │              │
                    │ Genera plan  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   EXECUTOR   │
                    │              │
                    │ Ejecuta E1   │
                    └──────┬───────┘
                           │
                           ▼
                  ¿Quedan pasos?
                     /        \
                   Sí          No
                   │            │
                   ▼            ▼
              EXECUTOR        FINAL
                   │            │
                   └─────┐      │
                         │      ▼
                         └──→  END


===========================================================
¿CUÁNDO UTILIZARLO?
===========================================================

Plan-and-Execute tiene sentido cuando podemos determinar
gran parte del workflow ANTES de comenzar a ejecutar.

Ejemplo:

    1. Buscar datos
    2. Generar dashboard
    3. Generar informe
    4. Generar documento
    5. Enviar email

El Planner puede decidir toda la secuencia al principio.

===========================================================
1. WORKFLOWS LARGOS Y ESTRUCTURADOS
===========================================================

Cuando una petición requiere varios pasos consecutivos.

Ejemplo:

    Buscar datos
        ↓
    Analizarlos
        ↓
    Generar gráfico
        ↓
    Generar PDF
        ↓
    Enviar email


===========================================================
2. AUTOMATIZACIÓN DE PROCESOS
===========================================================

Muy útil cuando el agente debe completar un proceso formado
por varias tareas conocidas.

Ejemplo:

    Buscar cliente
        ↓
    Buscar productos
        ↓
    Crear pedido
        ↓
    Generar documento
        ↓
    Enviar documento


===========================================================
3. CUANDO EL PLAN COMPLETO PUEDE GENERARSE AL PRINCIPIO
===========================================================

El Planner puede analizar la petición y decidir:

    E1 → buscar_datos
    E2 → generar_dashboard
    E3 → generar_informe
    E4 → enviar_email

sin tener que llamar al LLM después de cada herramienta.


===========================================================
4. REDUCIR LLAMADAS AL LLM
===========================================================

A diferencia de ReAct:

    LLM → Tool → LLM → Tool → LLM → Tool

Plan-and-Execute puede hacer:

    LLM Planner
          ↓
    Tool → Tool → Tool → Tool
          ↓
       LLM Final

Esto puede reducir llamadas al modelo.


===========================================================
5. PROCESOS DETERMINISTAS O SEMI-DETERMINISTAS
===========================================================

Es especialmente interesante cuando conocemos la estructura
general del proceso aunque los parámetros puedan cambiar.

Ejemplo:

    Buscar ventas
        ↓
    Generar dashboard
        ↓
    Generar informe


===========================================================
¿CUÁNDO NO ES LA MEJOR OPCIÓN?
===========================================================

No es ideal cuando no sabemos qué hacer después hasta
observar el resultado anterior.

Por ejemplo:

    Consultar SQL
        ↓
    Ver resultado
        ↓
    Decidir qué otra consulta hacer
        ↓
    Analizar resultado
        ↓
    Cambiar estrategia

En estos casos ReAct suele encajar mejor.


===========================================================
REGLA PRÁCTICA
===========================================================

Si puedes decir:

    "Antes de empezar puedo definir prácticamente todos
     los pasos que necesito ejecutar."

                → PLAN-AND-EXECUTE


Si puedes decir:

    "No sé qué herramienta necesitaré después hasta
     observar el resultado actual."

                → ReAct


===========================================================
RESUMEN
===========================================================

Usar Plan-and-Execute cuando:

    ✓ El proceso tiene varios pasos.
    ✓ Podemos generar un plan previamente.
    ✓ Las dependencias son relativamente conocidas.
    ✓ Queremos separar planificación y ejecución.
    ✓ Queremos reducir llamadas al LLM.
    ✓ El workflow es largo o estructurado.
    ✓ Queremos controlar mejor la ejecución.
    ✓ Queremos observar el plan completo.


ReAct cuando:

    ✓ La siguiente acción depende mucho del resultado anterior.
    ✓ El problema es exploratorio.
    ✓ El camino no se conoce de antemano.
    ✓ El agente necesita iterar y cambiar de estrategia.

===========================================================
"""

import os
import json
import time

from typing import TypedDict, Annotated
from operator import add

import pandas as pd
import uvicorn

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# =========================================================
# CONFIGURACIÓN
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)


app = FastAPI(
    title="Agente Plan-and-Execute",
    version="1.0.0"
)


# =========================================================
# ESTADO DEL AGENTE
# =========================================================

class AgentState(TypedDict):

    messages: Annotated[list, add_messages]

    input_tokens: Annotated[int, add]

    output_tokens: Annotated[int, add]

    llm_calls: Annotated[int, add]

    llm_latency: Annotated[float, add]

    trace: Annotated[list, add]

    # Plan generado por el Planner.
    plan: list[dict]

    # Índice del paso que se está ejecutando.
    paso_actual: int

    # Resultados de las herramientas.
    resultados: dict


# =========================================================
# HERRAMIENTAS
# =========================================================

def buscar_datos(
    columna: str = None,
    valor: str = None,
    limite: int = 10
):

    df = pd.read_csv("sales_data.csv")

    if columna and valor:

        if columna not in df.columns:
            raise ValueError(
                f"La columna '{columna}' no existe."
            )

        df = df[
            df[columna].astype(str).str.lower()
            == str(valor).lower()
        ]

    df = df.head(limite)

    return df.to_dict(orient="records")


def generar_dashboard(
    datos,
    columna_x: str = "product",
    columna_y: str = "revenue"
):

    df = pd.DataFrame(datos)

    # Aquí iría tu implementación real
    # de generación del dashboard.

    return {
        "archivo": "dashboard.png",
        "tipo": "dashboard",
        "filas": len(df)
    }


def generar_informe(
    datos,
    titulo: str = "Informe de ventas"
):

    df = pd.DataFrame(datos)

    # Aquí iría tu implementación real
    # de generación del PDF.

    return {
        "archivo": "informe.pdf",
        "tipo": "pdf",
        "titulo": titulo,
        "filas": len(df)
    }


def generar_documento(
    datos,
    titulo: str = "Documento de ventas"
):

    df = pd.DataFrame(datos)

    # Aquí iría tu implementación real
    # de generación del DOCX.

    return {
        "archivo": "documento.docx",
        "tipo": "docx",
        "titulo": titulo,
        "filas": len(df)
    }


def enviar_email(
    destinatario: str,
    asunto: str = "Informe"
):

    # Aquí iría el envío real del email.

    return {
        "estado": "enviado",
        "destinatario": destinatario,
        "asunto": asunto
    }


def descargar_documento(
    archivo: str
):

    return {
        "archivo": archivo,
        "estado": "disponible"
    }


# =========================================================
# MAPA DE HERRAMIENTAS
# =========================================================

TOOLS_MAP = {

    "buscar_datos":
        buscar_datos,

    "generar_dashboard":
        generar_dashboard,

    "generar_informe":
        generar_informe,

    "generar_documento":
        generar_documento,

    "enviar_email":
        enviar_email,

    "descargar_documento":
        descargar_documento
}


# =========================================================
# SCHEMA DEL PLAN
# =========================================================

class PasoPlan(BaseModel):

    herramienta: str = Field(
        description="Nombre de la herramienta que debe ejecutarse."
    )

    parametros: dict = Field(
        default_factory=dict,
        description="Parámetros de la herramienta."
    )


class Plan(BaseModel):

    pasos: list[PasoPlan] = Field(
        description="Pasos que debe ejecutar el agente en orden."
    )
# =========================================================
# PLANNER PROMPT
# =========================================================

planner_prompt = ChatPromptTemplate.from_template(
"""
Eres el Planner de un agente Plan-and-Execute.

Tu función es analizar la petición del usuario y generar
un plan completo de ejecución.

NO ejecutes herramientas.

NO respondas al usuario.

NO inventes resultados.

=========================================================
HERRAMIENTAS DISPONIBLES
=========================================================

1. buscar_datos

Descripción:
Busca datos en sales_data.csv.

Parámetros permitidos:

- columna: str | opcional
- valor: str | opcional
- limite: int | opcional

Ejemplo:

{{
    "columna": "product",
    "valor": "Laptop",
    "limite": 10
}}

IMPORTANTE:

NO existe ningún parámetro llamado:

- periodo
- fecha
- fechas
- categoria
- producto

Si el usuario pide información que no puede expresarse
con los parámetros disponibles, utiliza únicamente los
parámetros que realmente existan.

Si no se necesitan filtros:

{{
    "parametros": {{}}
}}


2. generar_dashboard

Parámetros permitidos:

- columna_x: str | opcional
- columna_y: str | opcional

IMPORTANTE:

NO incluyas "datos".

Los datos serán inyectados automáticamente
por el Executor desde buscar_datos.

Ejemplo:

{{
    "columna_x": "product",
    "columna_y": "revenue"
}}


3. generar_informe

Parámetros permitidos:

- titulo: str | opcional

IMPORTANTE:

NO incluyas "datos".

Los datos serán inyectados automáticamente.


4. generar_documento

Parámetros permitidos:

- titulo: str | opcional

IMPORTANTE:

NO incluyas "datos".

Los datos serán inyectados automáticamente.


5. enviar_email

Parámetros permitidos:

- destinatario: str
- asunto: str | opcional


6. descargar_documento

Parámetros permitidos:

- archivo: str


=========================================================
REGLAS DE PLANIFICACIÓN
=========================================================

1. buscar_datos debe ejecutarse antes de cualquier
   herramienta que necesite datos.

2. Las herramientas de generación reciben los datos
   automáticamente desde buscar_datos.

3. NO incluyas el parámetro "datos" en el plan.

4. enviar_email debe ejecutarse después de generar
   el archivo que debe enviarse.

5. descargar_documento debe ejecutarse después de
   generar el archivo correspondiente.

6. Utiliza únicamente parámetros que existan
   en la definición de cada herramienta.

7. NO inventes parámetros.

8. Si una herramienta no necesita parámetros,
   utiliza un diccionario vacío.

9. Utiliza valores por defecto cuando sea posible.

10. Genera todos los pasos necesarios antes
    de comenzar la ejecución.

11. Respeta el orden lógico de las dependencias.

12. Devuelve únicamente el plan estructurado.


=========================================================
PETICIÓN DEL USUARIO
=========================================================

{input}
"""
)

planner_chain = (
    planner_prompt
    | llm.with_structured_output(Plan)
)


# =========================================================
# PLANNER NODE
# =========================================================

def planner_node(state: AgentState):

    mensaje_usuario = state["messages"][-1].content

    inicio = time.perf_counter()

    plan = planner_chain.invoke({
        "input": mensaje_usuario
    })

    latencia = time.perf_counter() - inicio

    usage = getattr(
        plan,
        "usage_metadata",
        {}
    ) or {}

    pasos = [
        {
            "herramienta": paso.herramienta,
            "parametros": paso.parametros
        }
        for paso in plan.pasos
    ]

    return {

        "plan": pasos,

        "paso_actual": 0,

        "resultados": {},

        "llm_calls": 1,

        "llm_latency": latencia,

        "trace": [
            {
                "type": "planner",

                "latency": latencia,

                "plan": pasos
            }
        ]
    }


# =========================================================
# EXECUTOR NODE
# =========================================================

def executor_node(state: AgentState):

    plan = state["plan"]

    paso_actual = state["paso_actual"]

    resultados = state["resultados"].copy()


    # -----------------------------------------------------
    # Comprobar si quedan pasos
    # -----------------------------------------------------

    if paso_actual >= len(plan):

        return {}


    paso = plan[paso_actual]

    herramienta = paso["herramienta"]

    parametros = paso.get(
        "parametros",
        {}
    ).copy()


    inicio = time.perf_counter()


    # -----------------------------------------------------
    # INYECCIÓN DE RESULTADOS
    # -----------------------------------------------------

    if herramienta in [
        "generar_informe",
        "generar_dashboard",
        "generar_documento"
    ]:

        datos = resultados.get(
            "buscar_datos"
        )

        if datos is None:

            raise ValueError(
                "La herramienta requiere "
                "datos de buscar_datos."
            )

        parametros["datos"] = datos


    # -----------------------------------------------------
    # VALIDAR TOOL
    # -----------------------------------------------------

    if herramienta not in TOOLS_MAP:

        raise ValueError(
            f"Herramienta desconocida: {herramienta}"
        )


    funcion = TOOLS_MAP[herramienta]


    # -----------------------------------------------------
    # EJECUTAR TOOL
    # -----------------------------------------------------

    resultado = funcion(
        **parametros
    )


    latencia = (
        time.perf_counter()
        - inicio
    )


    # -----------------------------------------------------
    # GUARDAR RESULTADO
    # -----------------------------------------------------

    resultados[herramienta] = resultado


    # -----------------------------------------------------
    # TRACE
    # -----------------------------------------------------

    trace_item = {

        "type": "executor",

        "step": paso_actual + 1,

        "tool": herramienta,

        "parameters": parametros,

        "latency": latencia,

        "result": resultado
    }


    return {

        "paso_actual":
            paso_actual + 1,

        "resultados":
            resultados,

        "trace":
            [trace_item]
    }


# =========================================================
# FINAL NODE
# =========================================================

final_prompt = ChatPromptTemplate.from_template(
"""
Eres el asistente final de un agente Plan-and-Execute.

Petición original:

{input}

Resultados obtenidos:

{resultados}

Responde al usuario de forma clara y concisa.

Indica:

- qué se ha realizado
- qué archivos se han generado
- cualquier información relevante

No inventes información.

Utiliza únicamente los resultados proporcionados.
"""
)


final_chain = (
    final_prompt
    | llm
)


def final_node(state: AgentState):

    mensaje_usuario = (
        state["messages"][0].content
    )


    inicio = time.perf_counter()


    respuesta = final_chain.invoke({

        "input":
            mensaje_usuario,

        "resultados":
            json.dumps(
                state["resultados"],
                ensure_ascii=False,
                indent=2
            )
    })


    latencia = (
        time.perf_counter()
        - inicio
    )


    usage = getattr(
        respuesta,
        "usage_metadata",
        {}
    ) or {}


    input_tokens = usage.get(
        "input_tokens",
        0
    )

    output_tokens = usage.get(
        "output_tokens",
        0
    )


    return {

        "messages": [
            AIMessage(
                content=respuesta.content
            )
        ],

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "llm_calls":
            1,

        "llm_latency":
            latencia,

        "trace": [

            {
                "type": "final",

                "latency":
                    latencia,

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens
            }
        ]
    }


# =========================================================
# ROUTER
# =========================================================

def continuar_ejecucion(
    state: AgentState
):

    if (
        state["paso_actual"]
        <
        len(state["plan"])
    ):

        return "executor"


    return "final"


# =========================================================
# GRAFO
# =========================================================

workflow = StateGraph(
    AgentState
)


workflow.add_node(
    "planner",
    planner_node
)

workflow.add_node(
    "executor",
    executor_node
)

workflow.add_node(
    "final",
    final_node
)


workflow.add_edge(
    START,
    "planner"
)


workflow.add_edge(
    "planner",
    "executor"
)


workflow.add_conditional_edges(

    "executor",

    continuar_ejecucion,

    {
        "executor":
            "executor",

        "final":
            "final"
    }
)


workflow.add_edge(
    "final",
    END
)


memory = MemorySaver()


graph = workflow.compile(
    checkpointer=memory
)


# =========================================================
# API
# =========================================================

class AgentRequest(BaseModel):

    message: str

    thread_id: str = "default"


class AgentResponse(BaseModel):

    response: str


# =========================================================
# DATOS DEL DASHBOARD
# =========================================================

dashboard_data = {

    "llm_calls": 0,

    "input_tokens": 0,

    "output_tokens": 0,

    "total_tokens": 0,

    "llm_latency": 0.0,

    "tools": {},

    "plan": [],

    "trace": []
}


# =========================================================
# DASHBOARD
# =========================================================

def dashboard():

    # -----------------------------------------------------
    # TABLA DE HERRAMIENTAS
    # -----------------------------------------------------

    tools_html = ""


    for name, data in (
        dashboard_data["tools"].items()
    ):

        tools_html += f"""
        <tr>

            <td>
                {name}
            </td>

            <td>
                {data["calls"]}
            </td>

            <td>
                {data["latency"]:.3f}s
            </td>

        </tr>
        """


    # -----------------------------------------------------
    # PLAN
    # -----------------------------------------------------

    plan_html = ""


    for index, paso in enumerate(
        dashboard_data["plan"],
        start=1
    ):

        plan_html += f"""
        <div class="plan-item">

            <div class="plan-number">
                {index}
            </div>

            <div>

                <strong>
                    {paso["herramienta"]}
                </strong>

                <pre>
{json.dumps(
    paso["parametros"],
    ensure_ascii=False,
    indent=2
)}
                </pre>

            </div>

        </div>
        """


    # -----------------------------------------------------
    # TRACE
    # -----------------------------------------------------

    trace_html = ""


    for item in dashboard_data["trace"]:

        item_type = item["type"]


        # ================================================
        # PLANNER
        # ================================================

        if item_type == "planner":

            trace_html += f"""

            <div class="trace-item planner">

                <div class="trace-title">

                    🧠 PLANNER

                </div>

                <div>

                    Latencia:

                    <strong>
                        {item["latency"]:.3f}s
                    </strong>

                </div>

                <div>

                    Plan generado:

                    <strong>
                        {len(item["plan"])} pasos
                    </strong>

                </div>

            </div>

            """


        # ================================================
        # EXECUTOR
        # ================================================

        elif item_type == "executor":

            trace_html += f"""

            <div class="trace-item executor">

                <div class="trace-title">

                    🔧 EXECUTOR
                    —
                    Paso {item["step"]}

                </div>

                <div>

                    Tool:

                    <strong>
                        {item["tool"]}
                    </strong>

                </div>

                <div>

                    Latencia:

                    <strong>
                        {item["latency"]:.3f}s
                    </strong>

                </div>

                <div>

                    Parámetros:

                </div>

                <pre>
{json.dumps(
    item["parameters"],
    ensure_ascii=False,
    indent=2
)}
                </pre>

            </div>

            """


        # ================================================
        # FINAL
        # ================================================

        elif item_type == "final":

            trace_html += f"""

            <div class="trace-item final">

                <div class="trace-title">

                    💬 FINAL

                </div>

                <div>

                    Latencia:

                    <strong>
                        {item["latency"]:.3f}s
                    </strong>

                </div>

                <div>

                    Input tokens:

                    <strong>
                        {item["input_tokens"]}
                    </strong>

                </div>

                <div>

                    Output tokens:

                    <strong>
                        {item["output_tokens"]}
                    </strong>

                </div>

            </div>

            """


    # =====================================================
    # HTML
    # =====================================================

    return f"""
<!DOCTYPE html>

<html lang="es">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width,
             initial-scale=1.0"
>

<title>
    Plan-and-Execute Dashboard
</title>


<style>

* {{
    box-sizing: border-box;
}}


body {{

    margin: 0;

    padding: 30px;

    font-family:
        Arial,
        sans-serif;

    background:
        #f5f7fa;

    color:
        #1f2937;
}}


.dashboard {{

    max-width:
        1200px;

    margin:
        auto;
}}


h1 {{

    margin-bottom:
        30px;
}}


.metrics {{

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        20px;

    margin-bottom:
        30px;
}}


.card {{

    background:
        white;

    padding:
        25px;

    border-radius:
        12px;

    box-shadow:
        0 2px 10px
        rgba(0,0,0,0.08);
}}


.card-title {{

    color:
        #6b7280;

    font-size:
        14px;

    margin-bottom:
        10px;
}}


.card-value {{

    font-size:
        30px;

    font-weight:
        bold;
}}


.section {{

    background:
        white;

    padding:
        25px;

    border-radius:
        12px;

    margin-bottom:
        30px;

    box-shadow:
        0 2px 10px
        rgba(0,0,0,0.08);
}}


/* ======================================================
   PLAN
   ====================================================== */

.plan {{

    display:
        flex;

    flex-direction:
        column;

    gap:
        12px;
}}


.plan-item {{

    display:
        flex;

    gap:
        15px;

    padding:
        15px;

    background:
        #f8fafc;

    border-radius:
        8px;

    border-left:
        4px solid #6366f1;
}}


.plan-number {{

    min-width:
        32px;

    height:
        32px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        50%;

    background:
        #6366f1;

    color:
        white;

    font-weight:
        bold;
}}


pre {{

    background:
        #f3f4f6;

    padding:
        10px;

    border-radius:
        6px;

    overflow-x:
        auto;

    font-size:
        13px;
}}


/* ======================================================
   TABLA TOOLS
   ====================================================== */

table {{

    width:
        100%;

    border-collapse:
        collapse;
}}


th,
td {{

    text-align:
        left;

    padding:
        12px;

    border-bottom:
        1px solid #e5e7eb;
}}


th {{

    color:
        #6b7280;
}}


/* ======================================================
   TRACE
   ====================================================== */

.trace {{

    display:
        flex;

    flex-direction:
        column;

    gap:
        12px;
}}


.trace-item {{

    padding:
        15px;

    border-radius:
        8px;
}}


.planner {{

    background:
        #eef2ff;

    border-left:
        4px solid #6366f1;
}}


.executor {{

    background:
        #ecfdf5;

    border-left:
        4px solid #10b981;
}}


.final {{

    background:
        #fff7ed;

    border-left:
        4px solid #f97316;
}}


.trace-title {{

    font-weight:
        bold;

    margin-bottom:
        8px;
}}


@media(max-width: 800px) {{

    .metrics {{

        grid-template-columns:
            repeat(2, 1fr);
    }}

}}

</style>

</head>


<body>


<div class="dashboard">


<h1>

    🤖 Plan-and-Execute Agent Dashboard

</h1>


<!-- =====================================================
     MÉTRICAS
     ===================================================== -->

<div class="metrics">


<div class="card">

    <div class="card-title">
        Llamadas al LLM
    </div>

    <div class="card-value">

        {dashboard_data["llm_calls"]}

    </div>

</div>


<div class="card">

    <div class="card-title">
        Total Tokens
    </div>

    <div class="card-value">

        {dashboard_data["total_tokens"]}

    </div>

</div>


<div class="card">

    <div class="card-title">
        Latencia LLM
    </div>

    <div class="card-value">

        {dashboard_data["llm_latency"]:.2f}s

    </div>

</div>


<div class="card">

    <div class="card-title">
        Pasos del Plan
    </div>

    <div class="card-value">

        {len(dashboard_data["plan"])}

    </div>

</div>


</div>


<!-- =====================================================
     PLAN GENERADO
     ===================================================== -->

<div class="section">

<h2>
    🧠 Plan generado
</h2>


<div class="plan">

    {plan_html}

</div>

</div>


<!-- =====================================================
     HERRAMIENTAS
     ===================================================== -->

<div class="section">

<h2>
    🔧 Herramientas
</h2>


<table>

<thead>

<tr>

    <th>
        Herramienta
    </th>

    <th>
        Llamadas
    </th>

    <th>
        Latencia
    </th>

</tr>

</thead>


<tbody>

    {tools_html}

</tbody>

</table>

</div>


<!-- =====================================================
     TRACE
     ===================================================== -->

<div class="section">

<h2>
    🔍 Execution Trace
</h2>


<div class="trace">

    {trace_html}

</div>

</div>


</div>

</body>

</html>
"""


# =========================================================
# ENDPOINT /AGENT
# =========================================================

@app.post(
    "/agent",
    response_model=AgentResponse
)
def agent_endpoint(
    request: AgentRequest
):

    try:

        result = graph.invoke(

            {

                "messages": [

                    HumanMessage(
                        content=request.message
                    )

                ],

                "input_tokens": 0,

                "output_tokens": 0,

                "llm_calls": 0,

                "llm_latency": 0.0,

                "trace": [],

                "plan": [],

                "paso_actual": 0,

                "resultados": {}
            },

            config={
                "configurable": {
                    "thread_id":
                        request.thread_id
                }
            }
        )


        final_message = (
            result["messages"][-1]
        )


        # =================================================
        # ACTUALIZAR DASHBOARD
        # =================================================

        dashboard_data["llm_calls"] = (
            result["llm_calls"]
        )

        dashboard_data["input_tokens"] = (
            result["input_tokens"]
        )

        dashboard_data["output_tokens"] = (
            result["output_tokens"]
        )

        dashboard_data["total_tokens"] = (

            result["input_tokens"]

            +

            result["output_tokens"]
        )

        dashboard_data["llm_latency"] = (
            result["llm_latency"]
        )

        dashboard_data["plan"] = (
            result["plan"]
        )

        dashboard_data["trace"] = (
            result["trace"]
        )


        # =================================================
        # CONTABILIZAR TOOLS
        # =================================================

        dashboard_data["tools"] = {}


        for item in result["trace"]:

            if item["type"] != "executor":

                continue


            name = item["tool"]


            if name not in dashboard_data["tools"]:

                dashboard_data["tools"][name] = {

                    "calls": 0,

                    "latency": 0.0
                }


            dashboard_data["tools"][name]["calls"] += 1


            dashboard_data["tools"][name]["latency"] += (
                item["latency"]
            )


        return {

            "response":
                final_message.content
        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )


# =========================================================
# DASHBOARD ENDPOINT
# =========================================================

@app.get("/dashboard")
def get_dashboard():

    file_path = "dashboard.html"


    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            dashboard()
        )


    return FileResponse(

        path=file_path,

        media_type="text/html",

        filename=
            "plan_and_execute_dashboard.html"
    )


# =========================================================
# TEST
# =========================================================

@app.get("/test")
def test():

    return {

        "status":
            "ok"
    }


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8003
    )

"Ejemplo: Busca las ventas de la región Europe, genera un dashboard y un informe titulado \"Ventas Europa 2025\", y envíalo por email a ventas@empresa.com"