"""
===========================================================
AGENTE TREE OF THOUGHTS (ToT) - PLANIFICACIÓN DE REPARTO
===========================================================

                    Problema
                       │
                       ▼
                 cargar_datos
                       │
                       ▼
              generate_thoughts      (profundidad 1: ideas A, B, C)
                       │
                       ▼
              evaluate_thoughts      (Python determinista, CSV)
                       │
                  ¿profundidad < MAX?
                    /        \
                  Sí          No
                  │            │
                  ▼            ▼
          expand_thoughts     final
          (A1,A2,B1,B2...)      │
                  │             ▼
                  └──► evaluate END
                       (vuelve a evaluar)

El LLM propone (generate / expand / final).
Python valida y calcula distancia y coste (evaluate).
"""

import os
import json
import time
import html
import uuid

from functools import lru_cache
from typing import TypedDict, Annotated, Optional
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
    title="Agente Tree of Thoughts",
    version="1.0.0"
)

# Parámetros del árbol
MAX_PROFUNDIDAD = 2     # niveles del árbol (1 = ideas, 2 = hijos)
TOP_K = 3               # pensamientos que se expanden por nivel
HIJOS_POR_PADRE = 2     # hijos generados por cada pensamiento
MAX_FILAS_PROMPT = 60   # filas máximas por tabla enviadas al LLM


# =========================================================
# ESTADO DEL AGENTE
# =========================================================

class AgentState(TypedDict):

    messages: Annotated[list, add_messages]

    # Observabilidad
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    llm_calls: Annotated[int, add]
    llm_latency: Annotated[float, add]
    trace: Annotated[list, add]

    # Problema
    problema: str

    # Datos cargados (solo resumen; los DataFrames se leen
    # con get_datos() para que el estado sea serializable)
    resultados: dict

    # Árbol de pensamientos
    pensamientos: list[dict]            # nivel actual (sin evaluar)
    pensamientos_evaluados: list[dict]  # nivel actual (evaluados)
    mejores_pensamientos: list[dict]    # válidos del nivel, por coste
    arbol: list[dict]                   # todos los evaluados (todos los niveles)
    profundidad: int

    # Resultado
    mejor_solucion: Optional[dict]
    solucion: str


# =========================================================
# DATOS (CSV)
# =========================================================

@lru_cache(maxsize=1)
def get_datos() -> dict:
    """Carga los CSV una sola vez."""
    return {
        "pedidos": pd.read_csv("pedidos.csv"),
        "vehiculos": pd.read_csv("vehiculos.csv"),
        "clientes": pd.read_csv("clientes.csv"),
        "rutas": pd.read_csv("distancias.csv"),
    }


def resumir_datos(datos: dict) -> dict:
    """Resumen de los DataFrames para el prompt del LLM."""
    resumen = {}

    for nombre, df in datos.items():
        if isinstance(df, pd.DataFrame):
            resumen[nombre] = {
                "columnas": df.columns.tolist(),
                "filas": len(df),
                "datos": df.head(MAX_FILAS_PROMPT).to_dict(
                    orient="records"
                )
            }

    return resumen


# =========================================================
# SCHEMA DE PENSAMIENTOS
# =========================================================

class Pensamiento(BaseModel):

    id: str = Field(
        description="Identificador único del pensamiento."
    )

    parent_id: Optional[str] = Field(
        default=None,
        description="ID del pensamiento padre (null en el nivel 1)."
    )

    pensamiento: str = Field(
        description="Explicación breve de la estrategia."
    )

    id_vehiculo: str = Field(
        description="ID de un vehículo existente."
    )

    pedidos: list[str] = Field(
        description="IDs de pedidos existentes atendidos."
    )

    ruta: list[str] = Field(
        description=(
            "Secuencia ordenada de destinos que empieza y "
            "termina en 'Deposito'."
        )
    )

    profundidad: int = Field(
        default=1,
        description="Nivel del árbol."
    )


class Pensamientos(BaseModel):

    pensamientos: list[Pensamiento]


# =========================================================
# UTILIDAD: LLAMADA ESTRUCTURADA CON MÉTRICAS
# =========================================================

structured_llm = llm.with_structured_output(
    Pensamientos,
    include_raw=True
)


def invocar_estructurado(prompt: ChatPromptTemplate, variables: dict):
    """
    Invoca al LLM con salida estructurada y devuelve
    (pensamientos, input_tokens, output_tokens, latencia).
    """
    inicio = time.perf_counter()

    salida = (prompt | structured_llm).invoke(variables)

    latencia = time.perf_counter() - inicio

    parsed = salida.get("parsed")

    if parsed is None:
        raise ValueError(
            "El LLM no devolvió un JSON válido: "
            f"{salida.get('parsing_error')}"
        )

    uso = getattr(salida.get("raw"), "usage_metadata", None) or {}

    pensamientos = [p.model_dump() for p in parsed.pensamientos]

    return (
        pensamientos,
        uso.get("input_tokens", 0),
        uso.get("output_tokens", 0),
        latencia
    )


# =========================================================
# NODO: CARGAR DATOS
# =========================================================

def cargar_datos_node(state: AgentState) -> dict:

    inicio = time.perf_counter()

    datos = get_datos()

    resumen = {
        nombre: {
            "filas": len(df),
            "columnas": df.columns.tolist()
        }
        for nombre, df in datos.items()
    }

    return {
        "problema": state["messages"][-1].content,
        "resultados": resumen,
        "profundidad": 0,
        "pensamientos": [],
        "pensamientos_evaluados": [],
        "mejores_pensamientos": [],
        "arbol": [],               # se reinicia en cada ejecución
        "mejor_solucion": None,
        "solucion": "",
        "trace": [{
            "type": "load",
            "latency": time.perf_counter() - inicio,
            "tablas": resumen
        }]
    }


# =========================================================
# NODO: GENERAR PENSAMIENTOS (NIVEL 1)
# =========================================================

generate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres el generador de pensamientos de un agente Tree of Thoughts
especializado en planificación de reparto y logística.

Genera EXACTAMENTE 3 alternativas iniciales distintas para
resolver el problema. Cada una representa una estrategia
diferente de reparto (por ejemplo: minimizar distancia, usar
el vehículo más barato, agrupar por zonas...).

Reglas:
- Usa únicamente vehículos, pedidos y destinos que aparezcan
  en los datos. No inventes identificadores.
- No inventes distancias ni costes: se calcularán después con Python.
- Ten en cuenta la capacidad de peso y volumen del vehículo.
- Considera las ventanas horarias si existen.
- La ruta debe empezar y terminar en "Deposito". Si hay varios
  viajes, incluye "Deposito" entre ellos.
- Los pensamientos iniciales tienen parent_id null y profundidad 1.
- Ids: T1, T2, T3.
- No afirmes que una ruta es válida si no has comprobado todas
  sus restricciones.
"""
    ),
    (
        "human",
        """
Problema:
{problema}

Datos disponibles:
{datos}
"""
    )
])


def generate_thoughts(state: AgentState) -> dict:

    datos = get_datos()

    pensamientos, tin, tout, latencia = invocar_estructurado(
        generate_prompt,
        {
            "problema": state["problema"],
            "datos": json.dumps(
                resumir_datos(datos),
                ensure_ascii=False,
                default=str
            )
        }
    )

    # Forzar coherencia del nivel 1
    for i, p in enumerate(pensamientos, start=1):
        p["id"] = f"T{i}"
        p["parent_id"] = None
        p["profundidad"] = 1

    return {
        "pensamientos": pensamientos,
        "profundidad": 1,
        "input_tokens": tin,
        "output_tokens": tout,
        "llm_calls": 1,
        "llm_latency": latencia,
        "trace": [{
            "type": "generate",
            "latency": latencia,
            "pensamientos_generados": len(pensamientos),
            "input_tokens": tin,
            "output_tokens": tout
        }],
        "messages": [
            AIMessage(
                content=json.dumps(
                    pensamientos, ensure_ascii=False
                )
            )
        ]
    }


# =========================================================
# NODO: EVALUAR PENSAMIENTOS (DETERMINISTA)
# =========================================================

def evaluate_thoughts(state: AgentState) -> dict:
    """Valida las rutas y calcula peso, volumen, distancia y coste."""

    datos = get_datos()
    df_pedidos = datos["pedidos"]
    df_vehiculos = datos["vehiculos"]
    df_distancias = datos["rutas"]

    pensamientos = state.get("pensamientos", [])
    evaluados = []

    # Aceptar los nombres de columnas utilizados en el proyecto.
    col_pedido = (
        "pedido_id" if "pedido_id" in df_pedidos.columns
        else "id_pedido"
    )
    col_vehiculo = (
        "vehiculo_id" if "vehiculo_id" in df_vehiculos.columns
        else "id_vehiculo"
    )
    col_cap_peso = (
        "capacidad_peso_kg"
        if "capacidad_peso_kg" in df_vehiculos.columns
        else "capacidad_kg"
    )
    col_cap_volumen = "capacidad_volumen_l"
    col_coste = (
        "coste_por_km_eur"
        if "coste_por_km_eur" in df_vehiculos.columns
        else "coste_km"
    )

    matriz = df_distancias.set_index("origen")

    pedidos_por_id = {
        str(row[col_pedido]).strip(): row
        for _, row in df_pedidos.iterrows()
    }
    vehiculos_por_id = {
        str(row[col_vehiculo]).strip(): row
        for _, row in df_vehiculos.iterrows()
    }

    for pensamiento in pensamientos:
        errores = []
        evaluado = pensamiento.copy()

        id_vehiculo = str(
            pensamiento.get("id_vehiculo", "")
        ).strip()
        pedidos_ids = [
            str(p).strip()
            for p in pensamiento.get("pedidos", [])
        ]
        ruta = pensamiento.get("ruta", [])

        # 1. Validar vehículo y pedidos.
        vehiculo = vehiculos_por_id.get(id_vehiculo)
        if vehiculo is None:
            errores.append(f"El vehículo {id_vehiculo} no existe.")

        pedidos_seleccionados = []
        for pedido_id in pedidos_ids:
            pedido = pedidos_por_id.get(pedido_id)
            if pedido is None:
                errores.append(f"El pedido {pedido_id} no existe.")
            else:
                pedidos_seleccionados.append(pedido)

        if len(pedidos_ids) != len(set(pedidos_ids)):
            errores.append("Hay pedidos duplicados en la solución.")

        # 2. Validar la estructura de la ruta.
        if (
            not isinstance(ruta, list)
            or len(ruta) < 2
            or ruta[0] != "Deposito"
            or ruta[-1] != "Deposito"
        ):
            errores.append(
                "La ruta debe comenzar y terminar en Deposito."
            )
            ruta = ruta if isinstance(ruta, list) else []

        destinos_pedidos = [
            str(p["destino"]).strip()
            for p in pedidos_seleccionados
        ]
        destinos_ruta = [str(d).strip() for d in ruta]

        for destino in destinos_pedidos:
            if destino not in destinos_ruta:
                errores.append(
                    f"El destino {destino} no aparece en la ruta."
                )

        # 3. Calcular carga.
        peso_total = sum(
            float(p["peso_kg"]) for p in pedidos_seleccionados
        )
        volumen_total = sum(
            float(p["volumen_l"]) for p in pedidos_seleccionados
        )

        # 4. Comprobar capacidad de peso y volumen.
        if vehiculo is not None:
            capacidad_peso = float(vehiculo[col_cap_peso])
            capacidad_volumen = float(vehiculo[col_cap_volumen])

            if peso_total > capacidad_peso:
                errores.append(
                    f"Peso excedido: {peso_total} kg > "
                    f"{capacidad_peso} kg."
                )

            if volumen_total > capacidad_volumen:
                errores.append(
                    f"Volumen excedido: {volumen_total} l > "
                    f"{capacidad_volumen} l."
                )

        # 5. Calcular la distancia con la matriz, no con el LLM.
        distancia_total = 0.0

        for origen, destino in zip(ruta, ruta[1:]):
            if origen not in matriz.index:
                errores.append(
                    f"Origen desconocido en matriz: {origen}."
                )
                continue

            if destino not in matriz.columns:
                errores.append(
                    f"Destino desconocido en matriz: {destino}."
                )
                continue

            valor = matriz.loc[origen, destino]
            if isinstance(valor, (pd.Series, pd.DataFrame)):
                errores.append(
                    f"Tramo ambiguo en matriz: {origen} -> {destino}."
                )
                continue

            try:
                distancia_tramo = float(valor)
                if not pd.notna(distancia_tramo) or distancia_tramo < 0:
                    raise ValueError()
                distancia_total += distancia_tramo
            except (TypeError, ValueError):
                errores.append(
                    f"Distancia inválida para {origen} -> {destino}."
                )

        # 6. Calcular coste.
        coste_total = None
        if vehiculo is not None:
            coste_km = float(vehiculo[col_coste])
            if coste_km < 0:
                errores.append("El coste por kilómetro es negativo.")
            else:
                coste_total = round(distancia_total * coste_km, 2)

        evaluado["evaluacion"] = {
            "valido": len(errores) == 0,
            "errores": errores,
            "peso_total_kg": round(peso_total, 2),
            "volumen_total_l": round(volumen_total, 2),
            "distancia_calculada_km": round(distancia_total, 2),
            "coste_calculado_eur": coste_total,
        }
        evaluados.append(evaluado)

    # Ordenar solo las alternativas válidas por coste ascendente.
    validos = [
        p for p in evaluados
        if p["evaluacion"]["valido"]
        and p["evaluacion"]["coste_calculado_eur"] is not None
    ]
    validos.sort(
        key=lambda p: p["evaluacion"]["coste_calculado_eur"]
    )

    return {
        "pensamientos_evaluados": evaluados,
        "mejores_pensamientos": validos,
        "arbol": state.get("arbol", []) + evaluados,
        "trace": [{
            "type": "evaluate",
            "profundidad": state.get("profundidad", 1),
            "evaluados": len(evaluados),
            "validos": len(validos),
        }],
    }


# =========================================================
# NODO: EXPANDIR PENSAMIENTOS (NIVELES 2+)
# =========================================================

expand_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Eres el expansor de un agente Tree of Thoughts de logística
de reparto.

Recibes pensamientos padre ya evaluados con Python (con sus
errores, peso, volumen, distancia y coste reales). Para CADA
padre genera exactamente {hijos} pensamientos hijo que lo
mejoren:

- Si el padre es inválido, corrige sus errores concretos
  (capacidad, pedidos inexistentes, ruta mal formada...).
- Si el padre es válido, intenta reducir distancia y coste
  (reordenar paradas, cambiar de vehículo, reagrupar pedidos).

Reglas:
- Usa solo vehículos, pedidos y destinos que existan en los datos.
- No inventes distancias ni costes: se recalcularán con Python.
- La ruta empieza y termina en "Deposito".
- parent_id debe ser el id del padre del que procede cada hijo.
- profundidad = {profundidad}.
- Los hijos de un mismo padre deben ser distintos entre sí.
"""
    ),
    (
        "human",
        """
Problema:
{problema}

Datos disponibles:
{datos}

Pensamientos padre evaluados:
{padres}
"""
    )
])


def ordenar_para_expandir(evaluados: list[dict]) -> list[dict]:
    """Primero los válidos por coste, luego los inválidos con menos errores."""

    def clave(p):
        ev = p["evaluacion"]
        if ev["valido"] and ev["coste_calculado_eur"] is not None:
            return (0, ev["coste_calculado_eur"], 0)
        return (1, 0, len(ev["errores"]))

    return sorted(evaluados, key=clave)


def expand_thoughts(state: AgentState) -> dict:

    datos = get_datos()
    nueva_profundidad = state["profundidad"] + 1

    padres = ordenar_para_expandir(
        state["pensamientos_evaluados"]
    )[:TOP_K]

    padres_prompt = [
        {
            "id": p["id"],
            "pensamiento": p["pensamiento"],
            "id_vehiculo": p["id_vehiculo"],
            "pedidos": p["pedidos"],
            "ruta": p["ruta"],
            "evaluacion": p["evaluacion"],
        }
        for p in padres
    ]

    hijos, tin, tout, latencia = invocar_estructurado(
        expand_prompt,
        {
            "hijos": HIJOS_POR_PADRE,
            "profundidad": nueva_profundidad,
            "problema": state["problema"],
            "datos": json.dumps(
                resumir_datos(datos),
                ensure_ascii=False,
                default=str
            ),
            "padres": json.dumps(
                padres_prompt,
                ensure_ascii=False,
                default=str
            ),
        }
    )

    # Normalizar ids/parent_id para que el árbol sea coherente
    ids_padres = {p["id"] for p in padres}
    contador = {}

    for h in hijos:
        if h.get("parent_id") not in ids_padres:
            h["parent_id"] = padres[0]["id"]

        n = contador.get(h["parent_id"], 0) + 1
        contador[h["parent_id"]] = n

        h["id"] = f"{h['parent_id']}.{n}"
        h["profundidad"] = nueva_profundidad

    return {
        "pensamientos": hijos,
        "profundidad": nueva_profundidad,
        "input_tokens": tin,
        "output_tokens": tout,
        "llm_calls": 1,
        "llm_latency": latencia,
        "trace": [{
            "type": "expand",
            "profundidad": nueva_profundidad,
            "padres": [p["id"] for p in padres],
            "hijos_generados": len(hijos),
            "latency": latencia,
            "input_tokens": tin,
            "output_tokens": tout
        }],
    }


# =========================================================
# ROUTER
# =========================================================

def continuar_arbol(state: AgentState) -> str:

    if not state.get("pensamientos_evaluados"):
        return "final"

    if state["profundidad"] < MAX_PROFUNDIDAD:
        return "expand_thoughts"

    return "final"


# =========================================================
# NODO FINAL: ELEGIR LA MEJOR SOLUCIÓN
# =========================================================

final_prompt = ChatPromptTemplate.from_template(
"""
Eres el asistente final de un agente Tree of Thoughts de
planificación de reparto.

Petición original:
{input}

Mejor solución encontrada (validada y calculada con Python):
{mejor}

Resumen de todas las alternativas exploradas:
{alternativas}

Responde al usuario de forma clara y concisa:

- Indica la solución elegida (vehículo, pedidos y ruta).
- Indica distancia, coste, peso y volumen.
- Explica brevemente por qué es mejor que las demás.
- Si no hay ninguna solución válida, dilo y resume los errores.

Utiliza únicamente los datos proporcionados. No inventes información.
"""
)

final_chain = final_prompt | llm


def final_node(state: AgentState) -> dict:

    arbol = state.get("arbol", [])

    validos = [
        p for p in arbol
        if p["evaluacion"]["valido"]
        and p["evaluacion"]["coste_calculado_eur"] is not None
    ]
    validos.sort(
        key=lambda p: p["evaluacion"]["coste_calculado_eur"]
    )

    mejor = validos[0] if validos else None

    alternativas = [
        {
            "id": p["id"],
            "parent_id": p.get("parent_id"),
            "id_vehiculo": p["id_vehiculo"],
            "pedidos": p["pedidos"],
            "valido": p["evaluacion"]["valido"],
            "distancia_km": p["evaluacion"]["distancia_calculada_km"],
            "coste_eur": p["evaluacion"]["coste_calculado_eur"],
            "errores": p["evaluacion"]["errores"],
        }
        for p in arbol
    ]

    inicio = time.perf_counter()

    respuesta = final_chain.invoke({
        "input": state["problema"],
        "mejor": json.dumps(
            mejor, ensure_ascii=False, indent=2, default=str
        ) if mejor else "Ninguna solución válida.",
        "alternativas": json.dumps(
            alternativas, ensure_ascii=False, indent=2, default=str
        )
    })

    latencia = time.perf_counter() - inicio

    uso = getattr(respuesta, "usage_metadata", None) or {}
    tin = uso.get("input_tokens", 0)
    tout = uso.get("output_tokens", 0)

    return {
        "mejor_solucion": mejor,
        "solucion": respuesta.content,
        "messages": [AIMessage(content=respuesta.content)],
        "input_tokens": tin,
        "output_tokens": tout,
        "llm_calls": 1,
        "llm_latency": latencia,
        "trace": [{
            "type": "final",
            "latency": latencia,
            "input_tokens": tin,
            "output_tokens": tout,
            "mejor": mejor["id"] if mejor else None
        }]
    }


# =========================================================
# GRAFO
# =========================================================

workflow = StateGraph(AgentState)

workflow.add_node("cargar_datos", cargar_datos_node)
workflow.add_node("generate_thoughts", generate_thoughts)
workflow.add_node("evaluate_thoughts", evaluate_thoughts)
workflow.add_node("expand_thoughts", expand_thoughts)
workflow.add_node("final", final_node)

workflow.add_edge(START, "cargar_datos")
workflow.add_edge("cargar_datos", "generate_thoughts")
workflow.add_edge("generate_thoughts", "evaluate_thoughts")

workflow.add_conditional_edges(
    "evaluate_thoughts",
    continuar_arbol,
    {
        "expand_thoughts": "expand_thoughts",
        "final": "final"
    }
)

workflow.add_edge("expand_thoughts", "evaluate_thoughts")
workflow.add_edge("final", END)

memory = MemorySaver()

graph = workflow.compile(checkpointer=memory)


# =========================================================
# API
# =========================================================

class AgentRequest(BaseModel):

    message: str

    thread_id: str = "default"


class AgentResponse(BaseModel):

    response: str

    mejor_solucion: Optional[dict] = None


# =========================================================
# DATOS DEL DASHBOARD
# =========================================================

dashboard_data = {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "llm_latency": 0.0,
    "arbol": [],
    "mejor": None,
    "trace": []
}


# =========================================================
# DASHBOARD
# =========================================================

DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 30px; font-family: Arial, sans-serif;
       background: #f5f7fa; color: #1f2937; }
.dashboard { max-width: 1200px; margin: auto; }
h1 { margin-bottom: 30px; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr);
           gap: 20px; margin-bottom: 30px; }
.card, .section { background: white; padding: 25px; border-radius: 12px;
                  box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
.section { margin-bottom: 30px; }
.card-title { color: #6b7280; font-size: 14px; margin-bottom: 10px; }
.card-value { font-size: 30px; font-weight: bold; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 10px; border-bottom: 1px solid #e5e7eb;
         vertical-align: top; }
th { color: #6b7280; }
tr.best td { background: #ecfdf5; font-weight: bold; }
tr.invalid td { color: #9ca3af; }
.ok { color: #059669; font-weight: bold; }
.ko { color: #dc2626; font-weight: bold; }
pre { background: #f3f4f6; padding: 10px; border-radius: 6px;
      overflow-x: auto; font-size: 13px; }
.trace { display: flex; flex-direction: column; gap: 12px; }
.trace-item { padding: 15px; border-radius: 8px; }
.load { background: #f1f5f9; border-left: 4px solid #64748b; }
.generate { background: #eef2ff; border-left: 4px solid #6366f1; }
.expand { background: #fefce8; border-left: 4px solid #eab308; }
.evaluate { background: #ecfdf5; border-left: 4px solid #10b981; }
.final { background: #fff7ed; border-left: 4px solid #f97316; }
.trace-title { font-weight: bold; margin-bottom: 8px; }
@media(max-width: 800px) {
  .metrics { grid-template-columns: repeat(2, 1fr); }
}
"""


def esc(valor) -> str:
    return html.escape(str(valor))


def dashboard() -> str:

    mejor = dashboard_data["mejor"]
    mejor_id = mejor["id"] if mejor else None

    # -----------------------------------------------------
    # ÁRBOL DE PENSAMIENTOS
    # -----------------------------------------------------

    arbol_html = ""

    for p in sorted(
        dashboard_data["arbol"],
        key=lambda x: (x["profundidad"], x["id"])
    ):
        ev = p["evaluacion"]

        clase = (
            "best" if p["id"] == mejor_id
            else "" if ev["valido"]
            else "invalid"
        )

        estado = (
            '<span class="ok">✔ válido</span>' if ev["valido"]
            else '<span class="ko">✘ inválido</span>'
        )

        coste = (
            f'{ev["coste_calculado_eur"]:.2f} €'
            if ev["coste_calculado_eur"] is not None else "-"
        )

        errores = "<br>".join(esc(e) for e in ev["errores"]) or "-"

        arbol_html += f"""
        <tr class="{clase}">
            <td>{esc(p["id"])}</td>
            <td>{esc(p["profundidad"])}</td>
            <td>{esc(p.get("parent_id") or "-")}</td>
            <td>{esc(p["id_vehiculo"])}</td>
            <td>{esc(", ".join(p["pedidos"]))}</td>
            <td>{esc(" → ".join(p["ruta"]))}</td>
            <td>{ev["distancia_calculada_km"]} km</td>
            <td>{coste}</td>
            <td>{estado}</td>
            <td>{errores}</td>
        </tr>
        """

    # -----------------------------------------------------
    # MEJOR SOLUCIÓN
    # -----------------------------------------------------

    if mejor:
        mejor_html = f"<pre>{esc(json.dumps(mejor, ensure_ascii=False, indent=2, default=str))}</pre>"
    else:
        mejor_html = "<p>No se encontró ninguna solución válida.</p>"

    # -----------------------------------------------------
    # TRACE
    # -----------------------------------------------------

    trace_html = ""

    iconos = {
        "load": "📂 CARGAR DATOS",
        "generate": "🌱 GENERATE THOUGHTS",
        "expand": "🌿 EXPAND THOUGHTS",
        "evaluate": "🧮 EVALUATE THOUGHTS",
        "final": "🏁 FINAL",
    }

    for item in dashboard_data["trace"]:
        tipo = item.get("type", "")

        detalle = "".join(
            f"<div>{esc(k)}: <strong>{esc(v)}</strong></div>"
            for k, v in item.items()
            if k != "type" and not isinstance(v, (dict, list))
            or k in ("padres",)
        )

        trace_html += f"""
        <div class="trace-item {esc(tipo)}">
            <div class="trace-title">{iconos.get(tipo, esc(tipo))}</div>
            {detalle}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tree of Thoughts Dashboard</title>
<style>{DASHBOARD_CSS}</style>
</head>
<body>
<div class="dashboard">

<h1>🌳 Tree of Thoughts Agent Dashboard</h1>

<div class="metrics">
  <div class="card">
    <div class="card-title">Llamadas al LLM</div>
    <div class="card-value">{dashboard_data["llm_calls"]}</div>
  </div>
  <div class="card">
    <div class="card-title">Total Tokens</div>
    <div class="card-value">{dashboard_data["total_tokens"]}</div>
  </div>
  <div class="card">
    <div class="card-title">Latencia LLM</div>
    <div class="card-value">{dashboard_data["llm_latency"]:.2f}s</div>
  </div>
  <div class="card">
    <div class="card-title">Pensamientos evaluados</div>
    <div class="card-value">{len(dashboard_data["arbol"])}</div>
  </div>
</div>

<div class="section">
  <h2>🏆 Mejor solución</h2>
  {mejor_html}
</div>

<div class="section">
  <h2>🌳 Árbol de pensamientos</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Nivel</th><th>Padre</th><th>Vehículo</th>
        <th>Pedidos</th><th>Ruta</th><th>Distancia</th>
        <th>Coste</th><th>Estado</th><th>Errores</th>
      </tr>
    </thead>
    <tbody>{arbol_html}</tbody>
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

@app.post(
    "/agent",
    response_model=AgentResponse
)
def agent_endpoint(request: AgentRequest):

    try:

        # Un thread por ejecución: los contadores con reducer (add)
        # y el trace no se acumulan entre peticiones.
        thread = f"{request.thread_id}-{uuid.uuid4().hex[:8]}"

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=request.message)
                ],
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_calls": 0,
                "llm_latency": 0.0,
                "trace": [],
                "problema": request.message,
                "resultados": {},
                "pensamientos": [],
                "pensamientos_evaluados": [],
                "mejores_pensamientos": [],
                "arbol": [],
                "profundidad": 0,
                "mejor_solucion": None,
                "solucion": ""
            },
            config={
                "configurable": {"thread_id": thread}
            }
        )

        # =================================================
        # ACTUALIZAR DASHBOARD
        # =================================================

        dashboard_data["llm_calls"] = result["llm_calls"]
        dashboard_data["input_tokens"] = result["input_tokens"]
        dashboard_data["output_tokens"] = result["output_tokens"]
        dashboard_data["total_tokens"] = (
            result["input_tokens"] + result["output_tokens"]
        )
        dashboard_data["llm_latency"] = result["llm_latency"]
        dashboard_data["arbol"] = result["arbol"]
        dashboard_data["mejor"] = result["mejor_solucion"]
        dashboard_data["trace"] = result["trace"]

        return {
            "response": result["messages"][-1].content,
            "mejor_solucion": result["mejor_solucion"]
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

    file_path = "dashboard_tot.html"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dashboard())

    return FileResponse(
        path=file_path,
        media_type="text/html",
        filename="tot_dashboard.html"
    )


# =========================================================
# TEST
# =========================================================

@app.get("/test")
def test():

    return {"status": "ok"}


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8003
    )

"Ejemplo: Busca la mejor forma de repartir los pedidos P001 y P002"