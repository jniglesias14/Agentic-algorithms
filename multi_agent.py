import re
import sqlite3
from pathlib import Path

from langgraph.prebuilt import ToolNode, tools_condition
from starlette.responses import HTMLResponse

from react import tool_node

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
SQL_FILE = BASE_DIR / "pedidos.sql"
DB_FILE = BASE_DIR / "pedidos.db"

# Conectar con la base de datos
conexion = sqlite3.connect(DB_FILE)

try:
    # Leer el archivo SQL
    script_sql = SQL_FILE.read_text(encoding="utf-8")

    # Ejecutar el script: crear tabla e insertar registros
    conexion.executescript(script_sql)

    print("Tabla creada y registros insertados correctamente.")

    # Consultar los pedidos
    cursor = conexion.execute("SELECT * FROM pedidos")

    for pedido in cursor.fetchall():
        print(pedido)

finally:
    conexion.close()



import os
import json
import time
import html
import uuid

from functools import lru_cache
from typing import TypedDict, Annotated, Optional, Literal, Any
from operator import add

import pandas as pd
import uvicorn

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
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
    title="Multi Agent",
    version="1.0.0"
)

class AgentState(TypedDict):
    # Conversación compartida
    messages: Annotated[list, add_messages]

    # Enrutamiento
    next_agent: Literal[
        "tools_agent",
        "sql_analyst_agent",
        "response_agent",
        "END"
    ]

    # Agente SQL
    sql_query: str
    sql_valid: bool
    sql_error: str

    # Resultados de la base de datos
    query_results: list[dict[str, Any]]
    query_columns: list[str]

    # Agente de herramientas
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any

    # Control del flujo
    task_status: Literal[
        "pending",
        "running",
        "success",
        "error"
    ]
    error_message: str
    iterations: Annotated[int, add]

    # Observabilidad
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    llm_calls: Annotated[int, add]
    llm_latency: Annotated[float, add]



DB_PATH = Path(__file__).resolve().parent / "pedidos.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

@tool
def insertar_pedido(cliente: str, producto:str, cantidad:int, precio_unitario:float, fecha_pedido:str, estado:str="Pendiente"):
    """Inserta un pedido nuevo en la base de datos."""
    if not cliente.strip() or not producto.strip():
        return {"ok": False, "error": "Cliente y producto son obligatorios."}

    if cantidad <= 0 or precio_unitario < 0:
        return {"ok": False, "error": "Cantidad o precio no válidos."}

    try:
        with get_connection() as conn:
            cursor=conn.execute(
                """
                INSERT INTO pedidos(
                    cliente,producto,cantidad,precio_unitario,fecha_pedido,estado
                )
                VALUES(?,?,?,?,?,?)
                """,
                (
                    cliente, producto, cantidad,
                    precio_unitario, fecha_pedido, estado
                ),
            )
            return {
                "ok": True,
                "id_pedido": cursor.lastrowid,
                "mensaje": "Pedido insertado correctamente.",
            }
    except sqlite3.Error as exc:
        return {"ok": False, "error": str(exc)}


@tool
def actualizar_pedido(id_pedido:int,cliente:str,producto:str,cantidad:int,precio_unitario:float,fecha_pedido:str,estado:str):
    """Actualiza todos los campos editables de un pedido existente."""
    if not cliente.strip() or not producto.strip():
        return {"ok": False, "error": "Cliente y producto son obligatorios."}

    if cantidad <= 0 or precio_unitario < 0:
        return {"ok": False, "error": "Cantidad o precio no válidos."}

    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE pedidos
                SET cliente = ?,
                    producto = ?,
                    cantidad = ?,
                    precio_unitario = ?,
                    fecha_pedido = ?,
                    estado = ?
                WHERE id_pedido = ?
                """,
                (
                    cliente, producto, cantidad,
                    precio_unitario, fecha_pedido,
                    estado, id_pedido
                ),
            )

            if cursor.rowcount == 0:
                return {
                    "ok": False,
                    "mensaje": f"No existe el pedido {id_pedido}.",
                }

            return {
                "ok": True,
                "id_pedido": id_pedido,
                "mensaje": "Pedido actualizado correctamente.",
            }

    except sqlite3.Error as exc:

        return {"ok": False, "error": str(exc)}


@tool
def eliminar_pedido(id_pedido: int) -> dict:
    """Elimina un pedido por su identificador."""

    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM pedidos WHERE id_pedido = ?",
                (id_pedido,),
            )

            if cursor.rowcount == 0:
                return {
                    "ok": False,
                    "mensaje": f"No existe el pedido {id_pedido}.",
                }

            return {
                "ok": True,
                "id_pedido": id_pedido,
                "mensaje": "Pedido eliminado correctamente.",
            }

    except sqlite3.Error as exc:
        return {"ok": False, "error": str(exc)}


tools_pedidos=[
    insertar_pedido,
    actualizar_pedido,
    eliminar_pedido
]


from langchain_core.prompts import ChatPromptTemplate

router_prompt = ChatPromptTemplate.from_template("""
Eres el router de un sistema multiagente de gestión de pedidos.

DESTINOS DISPONIBLES:

1. tools_agent:
   Operaciones que modifican datos (insertar, actualizar, eliminar pedidos).

2. sql_analyst_agent:
   Consultas de solo lectura: listar, filtrar, contar, sumar, promediar,
   agrupar o comparar pedidos.

3. response_agent:
   Saludos, aclaraciones o preguntas generales que no requieren
   herramientas ni base de datos.

REGLAS:
- Selecciona un único destino.
- Una petición de consulta nunca es permiso para modificar registros.
- Si pide modificar y consultar, elige el destino de la primera acción.

PETICIÓN DEL USUARIO:
{user_input}

Devuelve únicamente una de estas palabras, sin comillas ni explicación:
tools_agent
sql_analyst_agent
response_agent
""")



agent_planner_prompt = ChatPromptTemplate.from_template("""
Eres un agente planner especializado en gestionar pedidos mediante
herramientas Python.

OBJETIVO:
Interpretar la petición del usuario, identificar las acciones
necesarias y ejecutar las herramientas adecuadas.

TOOLS DISPONIBLES:
- insertar_pedido:
  Inserta un pedido nuevo.
- actualizar_pedido:
  Actualiza los campos de un pedido existente.
- eliminar_pedido:
  Elimina un pedido por su identificador.

REGLAS DE EJECUCIÓN:

1. Analiza la intención del usuario antes de llamar a una herramienta.
2. Utiliza exclusivamente las tools disponibles para modificar pedidos.
3. No inventes identificadores, clientes, productos, cantidades,
   precios ni fechas.
4. Si faltan datos obligatorios, pregunta al usuario antes de ejecutar
   la operación.
5. Para actualizar un pedido, identifica el id_pedido correcto y
   proporciona todos los campos que exige la herramienta.
6. Antes de eliminar un pedido, solicita confirmación explícita
   del usuario si todavía no ha confirmado esa eliminación.
7. No afirmes que una operación se ha completado hasta recibir
   el resultado de la herramienta.
8. Si una herramienta devuelve un error, explícalo y no declares
   la operación como exitosa.
9. Si una operación tiene éxito, informa de los campos relevantes
   y del identificador del pedido.
10. No generes SQL directamente. Las operaciones de escritura deben
    realizarse mediante las tools autorizadas.

PLANIFICACIÓN:
- Determina qué operación necesita el usuario.
- Comprueba si tienes todos los datos necesarios.
- Ejecuta la herramienta correspondiente.
- Comprueba el resultado.
- Si la petición requiere varias acciones, ejecútalas en el orden
  correcto y verifica los resultados intermedios.

PETICIÓN DEL USUARIO:
{user_input}

CONTEXTO DISPONIBLE:
{messages}
""")



agent_sql_prompt = ChatPromptTemplate.from_template("""
Eres un agente analista SQL especializado en consultar una base
de datos SQLite que contiene información sobre pedidos.

ESQUEMA DE LA TABLA:

Tabla: pedidos

Columnas:
- id_pedido: INTEGER, identificador único del pedido.
- cliente: VARCHAR, nombre del cliente.
- producto: VARCHAR, nombre del producto.
- cantidad: INTEGER, unidades pedidas.
- precio_unitario: DECIMAL, precio por unidad.
- fecha_pedido: DATE, fecha del pedido.
- estado: VARCHAR, estado del pedido.

Estados habituales:
- Pendiente
- Enviado
- Entregado

OBJETIVO:
Traducir la petición del usuario a una consulta SQL correcta,
clara y eficiente.

REGLAS:
1. Genera exclusivamente consultas de lectura SELECT.
2. No generes INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   REPLACE ni ninguna operación que modifique datos.
3. Utiliza únicamente la tabla y las columnas especificadas.
4. No inventes columnas, tablas ni datos.
5. Para calcular el importe de un pedido utiliza:
   cantidad * precio_unitario.
6. Para calcular el importe total por cliente, utiliza SUM()
   y GROUP BY cliente.
7. Para filtrar resultados utiliza WHERE.
8. Para ordenar resultados utiliza ORDER BY.
9. Si el usuario solicita el total de pedidos, utiliza COUNT(*).
10. Si solicita ventas por cliente, agrupa por cliente.
11. Si falta información imprescindible para construir una consulta
    correcta, solicita una aclaración en lugar de inventar datos.
12. No ejecutes la consulta ni afirmes que has obtenido resultados.
13. No incluyas explicaciones, Markdown ni bloques de código
    en la salida SQL.

FORMATO DE SALIDA:
Devuelve únicamente la consulta SQL SELECT.

PETICIÓN DEL USUARIO:
{user_input}
""")

Destination=Literal[
    "tools_agent",
    "sql_analyst_agent",
    "response_agent",
]

def router_node(state: AgentState) -> dict:
    """Clasifica la petición y decide qué agente debe ejecutarla."""

    start = time.perf_counter()

    # Buscar el último mensaje del usuario
    user_input = next(
        (
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        "",
    )

    if not user_input:
        return {
            "next_agent": "response_agent",
            "error_message": "No se encontró un mensaje del usuario.",
        }

    response = (router_prompt | llm).invoke(
        {"user_input": user_input}
    )

    destination = str(response.content).strip()

    destination = destination.replace("`", "")

    # Validar el destino antes de enrutar
    valid_destinations = {
        "tools_agent",
        "sql_analyst_agent",
        "response_agent",
    }

    if destination not in valid_destinations:
        destination = "response_agent"

    return {
        "next_agent": destination,
        "llm_calls": 1,
        "llm_latency": time.perf_counter() - start,
        "iterations": 1,
    }


planner_model=llm.bind_tools(tools_pedidos)

def planner_node(state:AgentState):
    """Decide si debe llamar una herramienta o responder al usuario."""

    user_input = next(
        (
            m.content
            for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
        ),
        "",
    )
    response = (
            agent_planner_prompt | planner_model
    ).invoke({
        "user_input": user_input,
        "messages": state["messages"],
    })

    return {
        "messages": [response],
    }



def sql_node(state: AgentState) -> dict:
    """Genera, valida y ejecuta una consulta de solo lectura."""

    user_input = next(
        (
            m.content
            for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
        ),
        "",
    )

    try:
        response = (agent_sql_prompt | llm).invoke({
            "user_input": user_input,
        })

        sql = response.content.strip()
        sql = re.sub(r"^```sql\s*|\s*```$", "", sql).strip()

        # Validación inicial: una única consulta SELECT.
        if not re.match(r"^SELECT\b", sql, re.IGNORECASE):
            raise ValueError("Solo se permiten consultas SELECT.")

        # Rechazar consultas múltiples.
        if ";" in sql.rstrip(";"):
            raise ValueError("Solo se permite una consulta por petición.")

        # Abrir la base de datos en modo de solo lectura.
        db_uri = DB_PATH.resolve().as_uri() + "?mode=ro"

        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")

            cursor = conn.execute(sql)
            columns = [column[0] for column in cursor.description or []]
            rows = cursor.fetchall()

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        return {
            "sql_query": sql,
            "sql_valid": True,
            "sql_error": "",
            "query_results": results,
            "query_columns": columns,
            "task_status": "success",
            "messages": [
                AIMessage(
                    content=(
                        f"SQL ejecutada:\n{sql}\n\n"
                        f"Resultados:\n{results}"
                    )
                )
            ],
        }

    except (sqlite3.Error, ValueError) as exc:
        return {
            "sql_query": locals().get("sql", ""),
            "sql_valid": False,
            "sql_error": str(exc),
            "query_results": [],
            "query_columns": [],
            "task_status": "error",
            "error_message": str(exc),
            "messages": [
                AIMessage(
                    content=f"No se pudo ejecutar la consulta: {exc}"
                )
            ],
        }


def response_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
    }


tools_node = ToolNode(tools_pedidos)

builder = StateGraph(AgentState)
builder.add_node("router_node", router_node)
builder.add_node("planner_node", planner_node)
builder.add_node("tools_node", tools_node)
builder.add_node("sql_node", sql_node)
builder.add_node("response_node", response_node)

builder.add_edge(START, "router_node")


builder.add_conditional_edges(
    "router_node",
    lambda state: state["next_agent"],
    {
        "tools_agent": "planner_node",
        "sql_analyst_agent": "sql_node",
        "response_agent": "response_node",
    },
)

# El planner decide si necesita ejecutar una tool.
builder.add_conditional_edges(
    "planner_node",
    tools_condition,
    {
        "tools": "tools_node",
        END: END,
    },
)

# Tras ejecutar la herramienta, el planner interpreta su resultado.
builder.add_edge("tools_node", "planner_node")

builder.add_edge("sql_node", END)
builder.add_edge("response_node", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)



"""
config = {
    "configurable": {
        "thread_id": "prueba-1"
    }
}

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="¿Cuántos pedidos hay en total?"
            )
        ]
    },
    config=config,
)

print(result["messages"][-1].content)
"""


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
    "question": "",
    "route": "",
    "llm_calls": 0,
    "llm_latency": 0.0,
    "iterations": 0,
    "sql_query": "",
    "query_results": [],
    "tools": {},
    "trace": [],
}


# =========================================================
# DASHBOARD (HTML)
# =========================================================

def dashboard() -> str:
    e = html.escape

    tools_html = "".join(
        f"<tr><td>{e(name)}</td><td>{n}</td></tr>"
        for name, n in dashboard_data["tools"].items()
    ) or "<tr><td colspan='2'>Sin llamadas a herramientas</td></tr>"

    trace_html = ""
    for item in dashboard_data["trace"]:
        kind = item["type"]
        if kind == "router":
            body = f"🧭 ROUTER → <strong>{e(item['destination'])}</strong>"
        elif kind == "sql":
            body = (
                f"🗄️ SQL<pre>{e(item['query'])}</pre>"
                f"Filas devueltas: <strong>{item['rows']}</strong>"
            )
        elif kind == "tool_call":
            args = json.dumps(item["args"], ensure_ascii=False, indent=2)
            body = f"🔧 TOOL CALL: <strong>{e(item['tool'])}</strong><pre>{e(args)}</pre>"
        elif kind == "tool_result":
            body = f"📦 RESULTADO {e(item['tool'])}<pre>{e(item['content'])}</pre>"
        else:
            body = f"💬 RESPUESTA<pre>{e(item['content'])}</pre>"
        trace_html += f'<div class="trace-item {kind}">{body}</div>'

    results = dashboard_data["query_results"]
    results_json = json.dumps(results, ensure_ascii=False, indent=2, default=str)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Agent Dashboard</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:30px; font-family:Arial,sans-serif; background:#f5f7fa; color:#1f2937; }}
.dashboard {{ max-width:1100px; margin:auto; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:20px; margin-bottom:30px; }}
.card, .section {{ background:white; padding:25px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.08); }}
.section {{ margin-bottom:30px; }}
.card-title {{ color:#6b7280; font-size:14px; margin-bottom:10px; }}
.card-value {{ font-size:28px; font-weight:bold; }}
pre {{ background:#f3f4f6; padding:10px; border-radius:6px; overflow-x:auto; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ text-align:left; padding:12px; border-bottom:1px solid #e5e7eb; }}
th {{ color:#6b7280; }}
.trace {{ display:flex; flex-direction:column; gap:12px; }}
.trace-item {{ padding:15px; border-radius:8px; }}
.router {{ background:#eef2ff; border-left:4px solid #6366f1; }}
.sql {{ background:#f0f9ff; border-left:4px solid #0ea5e9; }}
.tool_call {{ background:#ecfdf5; border-left:4px solid #10b981; }}
.tool_result {{ background:#f7fee7; border-left:4px solid #84cc16; }}
.final {{ background:#fff7ed; border-left:4px solid #f97316; }}
@media(max-width:800px) {{ .metrics {{ grid-template-columns:repeat(2,1fr); }} }}
</style>
</head>
<body>
<div class="dashboard">
<h1>🤖 Multi-Agent Dashboard</h1>

<div class="metrics">
  <div class="card"><div class="card-title">Ruta</div>
    <div class="card-value" style="font-size:18px">{e(dashboard_data["route"] or "-")}</div></div>
  <div class="card"><div class="card-title">Llamadas al LLM</div>
    <div class="card-value">{dashboard_data["llm_calls"]}</div></div>
  <div class="card"><div class="card-title">Latencia router</div>
    <div class="card-value">{dashboard_data["llm_latency"]:.2f}s</div></div>
  <div class="card"><div class="card-title">Iteraciones</div>
    <div class="card-value">{dashboard_data["iterations"]}</div></div>
</div>

<div class="section">
  <h2>❓ Última petición</h2>
  <p>{e(dashboard_data["question"] or "-")}</p>
</div>

<div class="section">
  <h2>🗄️ Última consulta SQL</h2>
  <pre>{e(dashboard_data["sql_query"] or "No se ejecutó SQL en la última petición")}</pre>
  <pre>{e(results_json)}</pre>
</div>

<div class="section">
  <h2>🔧 Herramientas</h2>
  <table>
    <thead><tr><th>Herramienta</th><th>Llamadas</th></tr></thead>
    <tbody>{tools_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>🔍 Execution Trace</h2>
  <div class="trace">{trace_html}</div>
</div>

</div>
</body>
</html>
"""


# =========================================================
# ENDPOINT /AGENT
# =========================================================

@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(request: AgentRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # Métricas antes del turno (los reducers `add` acumulan entre turnos)
        before = graph.get_state(config).values or {}

        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )

        messages = result["messages"]
        final_message = messages[-1]
        route = result.get("next_agent", "")

        # Mensajes de ESTE turno: los posteriores al último HumanMessage
        last_human = max(
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        )
        turn_messages = messages[last_human + 1:]

        # -------------------------------------------------
        # TRACE + herramientas
        # -------------------------------------------------
        trace = [{"type": "router", "destination": route}]
        tools: dict[str, int] = {}

        if route == "sql_analyst_agent":
            trace.append({
                "type": "sql",
                "query": result.get("sql_query", ""),
                "rows": len(result.get("query_results", [])),
            })

        for m in turn_messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for call in m.tool_calls:
                    tools[call["name"]] = tools.get(call["name"], 0) + 1
                    trace.append({
                        "type": "tool_call",
                        "tool": call["name"],
                        "args": call["args"],
                    })
            elif isinstance(m, ToolMessage):
                trace.append({
                    "type": "tool_result",
                    "tool": m.name or "",
                    "content": str(m.content),
                })

        trace.append({"type": "final", "content": str(final_message.content)})

        # -------------------------------------------------
        # ACTUALIZAR DASHBOARD
        # -------------------------------------------------
        from_sql = route == "sql_analyst_agent"

        dashboard_data.update({
            "question": request.message,
            "route": route,
            "llm_calls": result.get("llm_calls", 0) - before.get("llm_calls", 0),
            "llm_latency": result.get("llm_latency", 0.0) - before.get("llm_latency", 0.0),
            "iterations": result.get("iterations", 0) - before.get("iterations", 0),
            "sql_query": result.get("sql_query", "") if from_sql else "",
            "query_results": result.get("query_results", []) if from_sql else [],
            "tools": tools,
            "trace": trace,
        })

        return {"response": str(final_message.content)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# =========================================================
# ENDPOINT /DASHBOARD (descarga el HTML)
# =========================================================

@app.get("/dashboard")
def get_dashboard():
    file_path = BASE_DIR / "dashboard.html"

    file_path.write_text(dashboard(), encoding="utf-8")

    return FileResponse(
        path=file_path,
        media_type="text/html",
        filename="multi_agent_dashboard.html",
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
if __name__ == "__main__":

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8003
    )


"Ejemplo: ¿Cuántos pedidos hay en total?"