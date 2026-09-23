"""A minimal local UI over the same evaluator used in the experiment."""
import json
import threading

import streamlit as st

from lab import MODEL_NAMES, ROOT, cases, load_agent, predict, read_json
from workflow_ui import FLOW_LABELS, clear_workflow_result, render_workflow

st.set_page_config(page_title="Laya · prueba local", page_icon="🧪", layout="wide")


@st.cache_resource(max_entries=1)
def engine(model):
    return load_agent("cpu", model), threading.Lock()


def clear_result():
    st.session_state.pop("last_result", None)
    clear_workflow_result()


st.title("Laya · prueba local")
st.caption("Tres variantes · Inferencia local · Soporte, guardrails, RAG y selección de modelo")
model_labels = {"multilingual": "Laya Multilingual · ES/PT/EN", "english": "Laya base · inglés", "typed-decisions": "Laya Typed-Decisions · inglés"}
model = st.selectbox("Modelo", MODEL_NAMES, format_func=model_labels.get, on_change=clear_result)
flow = st.selectbox("Experimento", list(FLOW_LABELS), format_func=FLOW_LABELS.get, on_change=clear_result)
if flow != "support":
    render_workflow(flow, model, model_labels[model], engine)
    st.stop()
if st.session_state.get("last_result", {}).get("model") != model:
    clear_result()
if model != "multilingual":
    st.caption("Esta variante está orientada a inglés. La comparación usa los mismos diez ejemplos ingleses.")
examples = cases("all" if model == "multilingual" else "en")
selection = st.selectbox("Ejemplo", range(len(examples)), format_func=lambda i: examples[i]["id"], key=f"example_{model}", on_change=clear_result)
text = st.text_area("Solicitud", value=examples[selection]["state"]["body"], height=130, key=f"input_{model}_{selection}")
with st.expander("Preguntas del experimento"):
    question_text = st.text_area("Preguntas JSON", json.dumps(read_json("questions.json"), ensure_ascii=False, indent=2), height=300)
if st.button("Analizar", type="primary"):
    try:
        if not text.strip():
            raise ValueError("Escribí una solicitud antes de analizar.")
        questions = json.loads(question_text)
        if not isinstance(questions, dict) or set(questions) != {"department", "urgency", "refund_requested"}:
            raise ValueError("Mantené los tres identificadores del experimento.")
        for key, expected_type in {"department": "choice", "urgency": "score", "refund_requested": "noul"}.items():
            if not isinstance(questions[key], dict) or questions[key].get("type") != expected_type:
                raise ValueError(f"La pregunta {key} debe usar el tipo {expected_type}.")
        with st.spinner("Cargando el modelo local y analizando…"):
            agent, lock = engine(model)
            with lock:
                result, latency = predict(agent, {"body": text}, questions)
        st.session_state["last_result"] = {"model": model, "model_lock": read_json("models.lock.json")[model], "input": text, "questions": questions, "result": result, "latency_ms": latency}
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        st.session_state.pop("last_result", None)
        st.error(str(error))

if "last_result" in st.session_state:
    saved = st.session_state["last_result"]
    answers = saved["result"]["answers"]
    st.caption("Resultado de: " + saved["input"])
    cols = st.columns(3)
    labels = {"billing": "Facturación", "technical": "Soporte técnico", "sales": "Ventas", "other": "Otros"}
    cols[0].metric("Departamento", labels.get(answers["department"]["choice"], answers["department"]["choice"]))
    cols[1].metric("Urgencia (0–2)", f"{answers['urgency']['score']:.2f}")
    cols[2].metric("Probabilidad de devolución", f"{100 * answers['refund_requested']['noul']:.1f} %")
    st.caption(f"{model_labels[saved['model']]} · CPU · {saved['latency_ms']:.1f} ms para las tres preguntas · carga inicial excluida")
    probabilities = answers["department"]["probabilities"]
    st.bar_chart({"Departamento": [labels.get(k, k) for k in probabilities], "Probabilidad": list(probabilities.values())}, x="Departamento", y="Probabilidad", horizontal=True)
    st.info("Las probabilidades son estimaciones del modelo. Contrastalas con los resultados del experimento; no garantizan acierto.")
    with st.expander("Respuesta original"):
        st.json(saved["result"])
    st.download_button("Descargar resultado JSON", json.dumps(saved, ensure_ascii=False, indent=2), "laya-result.json", "application/json")

summary_path = ROOT / "results/comparison-cpu/summary.json"
if summary_path.exists():
    with st.expander("Comparativa medida · los mismos diez ejemplos ingleses · CPU"):
        summary = json.loads(summary_path.read_text())
        st.table([{"Modelo": name, "Departamento": f"{m['department_correct']}/10", "Devolución": f"{m['refund_accuracy_threshold_0_5']:.0%}", "Urgencia MAE": round(m['urgency_mae_0_to_2'], 3), "Mediana ms": round(m['latency_p50_ms'], 1)} for name, m in summary["models"].items()])
st.caption("Laya 0.3.11 · checkpoint " + read_json("models.lock.json")[model]["revision"][:12] + " · resultados exploratorios, sin entrenamiento adicional")
