"""
===========================================================
¿CUÁNDO IMPLEMENTAR UN AGENTE ReAct?
===========================================================

ReAct = Reason + Act

El agente alterna entre:
    1. Razonar qué necesita hacer.
    2. Ejecutar una herramienta.
    3. Observar el resultado.
    4. Volver a razonar.
    5. Ejecutar otra herramienta si es necesario.
    6. Repetir hasta poder responder.



===========================================================
1. CONSULTAS DINÁMICAS
===========================================================

Tiene sentido ReAct cuando NO sabemos de antemano qué
herramientas tendrá que utilizar el agente.



===========================================================
2. ANÁLISIS DE DATOS
===========================================================


===========================================================
3. AGENTES SQL
===========================================================

Muy útil cuando tenemos un agente capaz de consultar
una base de datos.

===========================================================
4. CUANDO UNA TOOL DEPENDE DEL RESULTADO DE OTRA
===========================================================

===========================================================
5. ERP: CONSULTAS SOBRE DIFERENTES ENTIDADES
===========================================================

===========================================================
6. RESOLUCIÓN DE REFERENCIAS
===========================================================

===========================================================
7. AGENTES CON MUCHAS TOOLS
===========================================================

ReAct tiene sentido cuando tenemos muchas herramientas
y no queremos programar manualmente qué herramienta
debe utilizarse en cada caso.

===========================================================
8. INVESTIGACIÓN / EXPLORACIÓN
===========================================================

ReAct es especialmente útil cuando no sabemos de
antemano cuál será el camino para llegar a la respuesta.

===========================================================
9. CUANDO PUEDE SER NECESARIO REPETIR UNA ACCIÓN
===========================================================

===========================================================
10. CUANDO EL AGENTE NECESITA VERIFICAR INFORMACIÓN
===========================================================

===========================================================
¿CUÁNDO NO ES LA MEJOR OPCIÓN?
===========================================================

No todo debería ser ReAct.

Si tenemos un workflow conocido y determinista:
    1. Buscar cliente
    2. Buscar producto
    3. Crear pedido
    4. Generar documento
    5. Enviar email

Es decir, un patrón Plan-and-Execute.


===========================================================
RESUMEN
===========================================================

Usar ReAct cuando:
    ✓ La siguiente acción depende del resultado anterior.
    ✓ El problema es exploratorio.
    ✓ Hay múltiples herramientas disponibles.
    ✓ No conocemos de antemano el camino exacto.
    ✓ El agente necesita analizar datos.
    ✓ El agente trabaja con SQL.
    ✓ Puede necesitar varias consultas consecutivas.
    ✓ Puede necesitar cambiar de estrategia.
    ✓ Necesitamos iteración entre LLM y herramientas.

No usar ReAct necesariamente cuando:
    ✗ El workflow es completamente determinista.
    ✗ Los pasos son conocidos de antemano.
    ✗ Hay un proceso largo y estructurado.
    ✗ Necesitamos ejecutar siempre la misma secuencia.

===========================================================
REGLA PRÁCTICA
===========================================================

Si puedes decir:

    "No sé qué herramienta necesitaré después hasta
     ver qué me devuelve la herramienta actual."

                → ReAct


Si puedes decir:

    "Sé perfectamente los pasos que tengo que ejecutar
     antes de empezar."

                → Plan-and-Execute / Workflow
===========================================================


                 ┌─────────────┐
                 │             │
                 ▼             │
START → LLM → ¿tool_calls? ────┤
            │       │         │
            │ Sí    │ No      │
            ▼       ▼         │
          TOOLS    END        │
            │                  │
            └──────→ LLM ──────┘

"""

import os
import json
import time

from langchain_core.tools import tool
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import numpy as np
import uvicorn
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)


# Razonar → Actuar → Observar → Razonar → Actuar → ...


app = FastAPI(
    title="Agente ReAct",
    version="1.0.0"
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]

    llm_calls: Annotated[int, add]
    llm_latency: Annotated[float, add]

    trace: Annotated[list, add]


@tool
def read_csv(nombre: str = "sales_data.csv") -> str:
    """
    Carga el CSV y devuelve información básica sobre el dataset.
    """

    df = pd.read_csv(nombre)

    return json.dumps({
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records")
    }, ensure_ascii=False)


@tool
def analyze_sales(operation: str) -> str:
    """
    Analiza el dataset de ventas.

    Operaciones disponibles:
    - summary
    - total_revenue
    - total_profit
    - top_products
    - top_customers
    """

    df = pd.read_csv("sales_data.csv")

    if operation == "summary":

        return df.describe().to_json()

    elif operation == "total_revenue":

        return str(df["revenue"].sum())

    elif operation == "total_profit":

        return str(df["profit"].sum())

    elif operation == "top_products":

        result = (
            df.groupby("product")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        return result.to_json()

    elif operation == "top_customers":

        result = (
            df.groupby("customer")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        return result.to_json()

    return "Operación no válida"


llm_with_tools = llm.bind_tools([
    read_csv,
    analyze_sales
])


def call_llm(state: AgentState):

    start = time.perf_counter()

    response = llm_with_tools.invoke(
        state["messages"]
    )

    latency = time.perf_counter() - start

    usage = getattr(
        response,
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
        "messages": [response],

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,

        "llm_calls": 1,
        "llm_latency": latency,

        "trace": [{
            "type": "llm",
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }]
    }


# ===========================================================
# REGISTRO DE HERRAMIENTAS
# ===========================================================

class TrackedToolNode(ToolNode):

    def invoke(self, input, config=None, **kwargs):

        last_message = input["messages"][-1]

        tool_calls = getattr(
            last_message,
            "tool_calls",
            []
        )

        # Registramos cada herramienta ejecutada.
        # ToolNode sigue siendo el encargado de ejecutar
        # realmente las herramientas.
        start_times = {}

        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            start_times[tool_call_id] = time.perf_counter()

        result = super().invoke(
            input,
            config=config,
            **kwargs
        )

        trace = []

        for message in result.get("messages", []):

            tool_call_id = getattr(
                message,
                "tool_call_id",
                None
            )

            if tool_call_id is None:
                continue

            tool_name = None

            for tool_call in tool_calls:

                if tool_call["id"] == tool_call_id:
                    tool_name = tool_call["name"]
                    break

            start = start_times.get(
                tool_call_id
            )

            if start is not None:

                latency = (
                    time.perf_counter() - start
                )

                trace.append({
                    "type": "tool",
                    "name": tool_name,
                    "latency": latency,
                    "calls": 1
                })

        result["trace"] = trace

        return result


tool_node = TrackedToolNode([
    read_csv,
    analyze_sales
])


def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


graph = StateGraph(AgentState)


graph.add_node(
    "llm",
    call_llm
)

graph.add_node(
    "tools",
    tool_node
)


graph.add_edge(
    START,
    "llm"
)


graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)


graph.add_edge(
    "tools",
    "llm"
)


agent = graph.compile()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str



# ===========================================================
# DATOS DEL DASHBOARD
# ===========================================================

dashboard_data = {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "llm_latency": 0.0,
    "tools": {},
    "trace": []
}

# ===========================================================
# DASHBOARD
# ===========================================================

def dashboard():

    tools_html = ""

    for name, data in dashboard_data["tools"].items():

        tools_html += f"""
        <tr>
            <td>{name}</td>
            <td>{data["calls"]}</td>
            <td>{data["latency"]:.3f}s</td>
        </tr>
        """

    trace_html = ""

    for item in dashboard_data["trace"]:

        if item["type"] == "llm":

            trace_html += f"""
            <div class="trace-item llm">
                <div class="trace-title">
                    🤖 LLM
                </div>

                <div>
                    Latencia:
                    <strong>{item["latency"]:.3f}s</strong>
                </div>

                <div>
                    Input tokens:
                    <strong>{item["input_tokens"]}</strong>
                </div>

                <div>
                    Output tokens:
                    <strong>{item["output_tokens"]}</strong>
                </div>
            </div>
            """

        elif item["type"] == "tool":

            trace_html += f"""
            <div class="trace-item tool">
                <div class="trace-title">
                    🔧 TOOL: {item["name"]}
                </div>

                <div>
                    Latencia:
                    <strong>{item["latency"]:.3f}s</strong>
                </div>
            </div>
            """

    return f"""
    <!DOCTYPE html>

    <html lang="es">

    <head>

        <meta charset="UTF-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <title>ReAct Dashboard</title>

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

                background: #f5f7fa;
                color: #1f2937;
            }}

            h1 {{
                margin-bottom: 30px;
            }}

            .dashboard {{
                max-width: 1200px;
                margin: auto;
            }}

            .metrics {{
                display: grid;

                grid-template-columns:
                    repeat(4, 1fr);

                gap: 20px;

                margin-bottom: 30px;
            }}

            .card {{
                background: white;

                padding: 25px;

                border-radius: 12px;

                box-shadow:
                    0 2px 10px
                    rgba(0,0,0,0.08);
            }}

            .card-title {{
                color: #6b7280;

                font-size: 14px;

                margin-bottom: 10px;
            }}

            .card-value {{
                font-size: 30px;

                font-weight: bold;
            }}

            .section {{
                background: white;

                padding: 25px;

                border-radius: 12px;

                margin-bottom: 30px;

                box-shadow:
                    0 2px 10px
                    rgba(0,0,0,0.08);
            }}

            table {{
                width: 100%;

                border-collapse:
                    collapse;
            }}

            th,
            td {{
                text-align: left;

                padding: 12px;

                border-bottom:
                    1px solid #e5e7eb;
            }}

            th {{
                color: #6b7280;
            }}

            .trace {{
                display: flex;

                flex-direction: column;

                gap: 12px;
            }}

            .trace-item {{
                padding: 15px;

                border-radius: 8px;
            }}

            .llm {{
                background: #eef2ff;

                border-left:
                    4px solid #6366f1;
            }}

            .tool {{
                background: #ecfdf5;

                border-left:
                    4px solid #10b981;
            }}

            .trace-title {{
                font-weight: bold;

                margin-bottom: 8px;
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
                🤖 ReAct Agent Dashboard
            </h1>


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
                        Herramientas utilizadas
                    </div>

                    <div class="card-value">
                        {len(dashboard_data["tools"])}
                    </div>

                </div>

            </div>


            <div class="section">

                <h2>
                    🔧 Herramientas
                </h2>

                <table>

                    <thead>

                        <tr>
                            <th>Herramienta</th>
                            <th>Llamadas</th>
                            <th>Latencia</th>
                        </tr>

                    </thead>

                    <tbody>

                        {tools_html}

                    </tbody>

                </table>

            </div>


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

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        result = agent.invoke({

            "messages": [
                HumanMessage(
                    content=request.message
                )
            ],

            "input_tokens": 0,
            "output_tokens": 0,

            "llm_calls": 0,
            "llm_latency": 0.0,

            "trace": []
        })


        final_message = result["messages"][-1]


        # ===================================================
        # ACTUALIZAR DASHBOARD
        # ===================================================

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

        dashboard_data["trace"] = (
            result["trace"]
        )


        # ===================================================
        # CONTABILIZAR TOOLS
        # ===================================================

        dashboard_data["tools"] = {}

        for item in result["trace"]:

            if item["type"] != "tool":
                continue

            name = item["name"]

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
            "response": final_message.content
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/dashboard")
def get_dashboard():
    file_path = "dashboard.html"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dashboard())

    return FileResponse(
        path=file_path,
        media_type="text/html",
        filename="agent_dashboard.html"
    )

@app.get("/test")
def test():

    print(">>> LLEGÓ A TEST")

    return {
        "status": "ok"
    }


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8003
    )

"Ejemplo: Dame los 3 mejores clientes y los 3 mejores productos por ingresos, y dime si hay algún producto que coincida con lo que compran esos clientes"