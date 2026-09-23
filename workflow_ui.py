"""Interactive inputs use the exact questions and decision rules of the experiment."""
import json

import streamlit as st

from lab import ROOT, read_json
from workflows import run_case

FLOW_LABELS = {"support": "Soporte · prueba original", "guardrails": "Guardrails", "rag": "Filtro RAG", "routing": "Selección de modelo"}
ACTION_LABELS = {"BLOCK": "Bloquear", "REVIEW": "Revisión humana", "PASS": "Permitir",
                 "KEEP": "Conservar", "DROP_INJECTION": "Descartar: instrucciones maliciosas",
                 "DROP_IRRELEVANT": "Descartar: irrelevante", "REVIEW_CONTRADICTION": "Revisar contradicción",
                 "SMALL": "Modelo pequeño", "LARGE": "Modelo grande", "TOOLS": "Ruta con herramientas"}


def clear_workflow_result():
    st.session_state.pop("workflow_result", None)


def render_workflow(flow, model, model_label, engine):
    config = read_json("workflows.json")[flow]
    descriptions = {
        "guardrails": "Detecta intentos de manipulación y datos sensibles. Los ejemplos incluyen ataques citados sin intención de ejecutarlos.",
        "rag": "Evalúa un fragmento frente a una consulta y una referencia de confianza: relevancia, instrucciones ocultas y contradicciones.",
        "routing": "Propone modelo pequeño, grande o herramientas según una política fija. La propuesta se muestra aquí; no llama a ningún proveedor."}
    st.subheader(config["title"])
    st.caption(descriptions[flow] + " · Casos de prueba en inglés")
    saved = st.session_state.get("workflow_result", {})
    if saved.get("model") != model or saved.get("workflow") != flow:
        clear_workflow_result()
    examples = [c for c in read_json("workflow_cases.json") if c["workflow"] == flow]
    choice = st.selectbox("Ejemplo", range(len(examples)), format_func=lambda i: examples[i]["id"],
                          key="workflow_example_" + flow, on_change=clear_workflow_result)
    labels = {"prompt": "Texto a comprobar", "query": "Consulta", "reference": "Referencia de confianza", "passage": "Fragmento recuperado", "request": "Petición"}
    state = {field: st.text_area(labels[field], value=examples[choice]["state"][field],
                key=f"workflow_input_{flow}_{choice}_{field}", height=85 if flow == "rag" else 130,
                on_change=clear_workflow_result) for field in config["fields"]}
    with st.expander("Preguntas y umbrales del experimento"):
        st.json({"questions": config["questions"], "thresholds": config["thresholds"]})
    if st.button("Analizar", type="primary", key="workflow_analyze"):
        try:
            from workflows import validate_state
            validate_state(flow, state)
            with st.spinner("Analizando localmente…"):
                agent, lock = engine(model)
                with lock:
                    result = run_case(agent, flow, state)
            st.session_state["workflow_result"] = {"workflow": flow, "model": model,
                "model_lock": read_json("models.lock.json")[model], "state": state, **result}
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            clear_workflow_result()
            st.error(str(error))
    if "workflow_result" in st.session_state:
        saved = st.session_state["workflow_result"]
        st.metric("Decisión propuesta", ACTION_LABELS[saved["action"]])
        st.caption(f"{model_label} · CPU · {saved['latency_ms']:.1f} ms · carga inicial excluida")
        st.table([{"Señal": q, "Valor": round(a.get("noul", a.get("score", 0)), 3)}
                  for q, a in saved["result"]["answers"].items()])
        if saved["state"] == examples[choice]["state"]:
            expected = examples[choice]["expected"]["action"]
            st.caption("Etiqueta fijada del ejemplo: " + ACTION_LABELS[expected] +
                       (" · coincide" if saved["action"] == expected else " · no coincide"))
        with st.expander("Respuesta original"):
            st.json(saved["result"])
        st.download_button("Descargar resultado JSON", json.dumps(saved, ensure_ascii=False, indent=2),
                           f"laya-{flow}.json", "application/json")
    summary_file = ROOT / "results/workflows-cpu/summary.json"
    if summary_file.exists():
        with st.expander("Comparación medida · 12 casos ingleses por modelo"):
            summary = json.loads(summary_file.read_text())
            st.table([{"Modelo": name, "Decisiones correctas": f"{data[flow]['action_correct']}/12",
                       "Mediana CPU ms": round(data[flow]["latency_p50_ms"], 1)}
                       for name, data in summary["models"].items()])
    st.caption("Prueba exploratoria con datos sintéticos y reglas fijas. Las probabilidades y decisiones pueden fallar.")
