# Resultado de la prueba — 23 de septiembre de 2026

**Laya funcionó localmente y fue rápido en GPU. Con estas preguntas y este conjunto pequeño, no fue suficientemente fiable para automatizar el flujo evaluado.**

## Qué se ejecutó

MacBook Pro M1 Max, 32 GB; Python 3.12.8, Laya 0.3.11, PyTorch 2.14.0 y Transformers 5.17.0. Checkpoint `convaiinnovations/laya/multilingual`, revisión `aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8`. Se utilizó el mismo modelo para los tres idiomas, sin router ni entrenamiento adicional.

Primero se hicieron tres pruebas de humo. Después se fijaron diez situaciones sintéticas con traducciones ES/PT/EN: **30 entradas, no 30 situaciones independientes**. Preguntas y etiquetas quedaron fijadas antes de evaluar, sin ajustes posteriores para mejorar las cifras. Los textos y etiquetas se prepararon con asistencia de Codex; no son un benchmark externo ni una muestra de clientes.

Cada solicitud contesta tres preguntas: departamento (cuatro categorías), urgencia (puntuación esperada entre 0 y 2) y probabilidad de devolución. Las instrucciones están en inglés. La rúbrica distingue petición rutinaria, plazo con actividad posible y actividad bloqueada. En mensajes mixtos se etiqueta la petición actual.

CPU y MPS se midieron secuencialmente, con cuatro hilos de PyTorch, tres solicitudes de calentamiento y una ejecución por entrada. El tiempo incluye las tres preguntas del SDK, excluye carga y prevalidación; MPS se sincroniza antes y después. Son mediciones de esta ejecución, no garantías de rendimiento.

## Cifras

| Medida | CPU | GPU / MPS |
|---|---:|---:|
| Departamento correcto | 18/30 · 60 % | 18/30 · 60 % |
| Error absoluto medio de urgencia (escala 0–2) | 0,815 | 0,815 |
| Devolución correcta, umbral 0,5 | 18/30 · 60 % | 18/30 · 60 % |
| Brier de devolución; menor es mejor | 0,259 | 0,259 |
| Latencia p50, tres preguntas | 126,2 ms | 33,4 ms |
| Latencia p95 | 159,3 ms | 41,4 ms |
| Carga con archivos en disco, incluidos imports | 4,77 s | 5,19 s |

La GPU fue aproximadamente **3,8 veces más rápida en la mediana**, sin cambiar ninguna categoría. La máxima diferencia CPU/MPS en urgencia fue 0,0001.

En departamento: español 5/10, portugués 7/10 e inglés 6/10. Elegir siempre una categoría mayoritaria daría 30 %. En devolución hay solo seis positivos: responder siempre «no» daría **80 %**, superior al 60 % del modelo. La urgencia tuvo un error de 0,815 puntos; las etiquetas dependen de nuestra rúbrica.

## Errores que importan

- Una caída del botón de pago se asignó a facturación en los tres idiomas, aunque se esperaba soporte técnico. En español, la probabilidad de esa categoría fue 1,0000, redondeada por el SDK.
- «No quiero una devolución» recibió una probabilidad de devolución de **87,0 %** en español; falló también en portugués e inglés.
- Una consulta de precios sin fecha límite recibió urgencia **1,64/2**, aunque la etiqueta era 0.
- Ninguna de las seis entradas etiquetadas como «otros» obtuvo esa categoría.

Una probabilidad alta no equivale a acierto ni demuestra calibración. El campo `confidence` del SDK se conserva sin reinterpretarlo como precisión observada.

## Reproducción y límites

Se creó otro entorno desde `uv.lock`, reutilizando paquetes y checkpoint en caché. macOS bloqueó las conexiones IPv4 e IPv6 del proceso; los 30 casos terminaron y **sus respuestas JSON fueron idénticas a CPU**. Pasaron cuatro tests y la demostración se comprobó con una inferencia real y una entrada vacía.

La instalación inicial sí necesitó red. La repetición no volvió a descargar los pesos. Los JSON publicados conservan las respuestas, métricas, versiones y hashes; los tiempos varían entre ejecuciones.

El conjunto es pequeño, sintético y relacionado por traducciones; no permite generalizar ni ordenar idiomas por calidad. No se compararon otros checkpoints, instrucciones en español, entrenamiento fino ni otros modelos. Un resultado peor no demuestra que Laya falle en cualquier aplicación; delimita esta configuración concreta.

Fuentes primarias: [SDK](https://github.com/NandhaKishorM/laya), [modelo](https://huggingface.co/convaiinnovations/laya). Evidencia: [CPU](results/cpu.json), [MPS](results/mps.json), [repetición offline](results/cpu-repro-offline.json), [verificación](results/verification.json).
