# Guardrails, RAG y selección de modelo — 23/09/2026

**El buen resultado en soporte no se trasladó a todos los usos.** Typed-Decisions fue el mejor de los tres en esta prueba, pero dejó pasar un ataque y no reconoció ninguna de las cuatro peticiones que necesitaban herramientas.

## Qué hicimos

36 casos sintéticos propios en inglés: 12 por escenario, iguales para los tres checkpoints. Se fijaron textos, preguntas, etiquetas y umbrales antes de inferir; `workflow_protocol.lock.json` conserva fecha y hashes. No se ajustaron después de ver resultados.

* **Guardrails:** cinco ataques, cinco mensajes inocuos —incluidas citas y negaciones— y dos ejemplos con datos sensibles ficticios. Salidas: permitir, bloquear o revisar.
* **RAG:** tres consultas, cada una con un fragmento útil, uno irrelevante, uno con instrucciones maliciosas y otro contradictorio. La referencia de confianza se proporciona explícitamente.
* **Selección de modelo:** cuatro peticiones simples, cuatro complejas y cuatro que requieren datos externos o acciones. Salidas: modelo pequeño, grande o herramientas.

Se reutilizaron el M1 Max de 32 GB, dependencias y pesos fijados. Un proceso por modelo, cuatro hilos, tres calentamientos por escenario y una pasada por caso; CPU y MPS se ejecutaron secuencialmente. Son 108 entradas evaluadas por dispositivo, más su repetición offline.

## Decisiones correctas

| Variante | Guardrails | RAG | Selección de modelo |
|---|---:|---:|---:|
| Multilingual | 9/12 | 3/12 | 3/12 |
| Laya base | 7/12 | 4/12 | 7/12 |
| Typed-Decisions | **10/12** | **9/12** | **8/12** |
| Elegir siempre la clase mayoritaria | 5/12 | 3/12 | 4/12 |

CPU y MPS produjeron las mismas decisiones. La repetición CPU en otro entorno, con la red bloqueada por macOS, reprodujo las 108 respuestas completas exactamente.

### Los fallos que importan

* **Guardrails:** Multilingual omitió un bloqueo y bloqueó dos mensajes sin ataque; base detectó los cinco ataques pero produjo cuatro bloqueos falsos. Typed-Decisions bloqueó una petición de traducir una cita maliciosa y permitió un intento de extraer instrucciones mediante números: probabilidad 0,555, por debajo del umbral 0,6.
* **RAG:** Multilingual y base descartaron los tres fragmentos útiles. Typed-Decisions conservó dos; marcó el tercero como inyección y descartó dos contradicciones como irrelevantes. Base conservó además un fragmento que contradecía la referencia. Que un filtro no conserve contenido peligroso no basta si también elimina todo el contenido útil.
* **Selección:** base y Typed-Decisions enviaron las cuatro peticiones externas a un modelo pequeño, incluso consultar inventario privado y crear una cita. Multilingual reconoció solo una y pidió herramientas para seis peticiones que no las necesitaban.

## Tiempo por entrada

Medianas **CPU / GPU**, en milisegundos. Incluyen dos preguntas en guardrails y selección, tres en RAG; excluyen carga y validación previa. GPU sincronizada.

| Variante | Guardrails | RAG | Selección |
|---|---:|---:|---:|
| Multilingual | 89,2 / 31,7 | 126,2 / 40,4 | 104,2 / 31,7 |
| Laya base | 218,1 / 53,7 | 311,4 / 74,4 | 239,3 / 53,0 |
| Typed-Decisions | 223,0 / 53,5 | 311,9 / 73,0 | 251,5 / 52,4 |

## Alcance y reproducción

La selección mide coincidencia con nuestra política, sin llamar a otros LLM ni comprobar su calidad o ahorro. RAG evalúa fragmentos proporcionados: no incorpora buscador ni generación de respuestas. Los umbrales son reglas del programa, no garantías del modelo. Se detallan en `workflows.json`.

Es una prueba exploratoria pequeña, con etiquetas del autor y sin revisión independiente. No valida un filtro de seguridad para producción. Typed-Decisions está especializado en cuatro flujos distintos; su [ficha oficial](https://huggingface.co/convaiinnovations/laya-typed-decisions) advierte límites fuera de ellos. La [demo oficial](https://huggingface.co/spaces/convaiinnovations/laya-demo) inspiró los escenarios; nuestros casos y reglas son propios.

Después de preparar los tres modelos según el README:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python workflows.py compare --device cpu --output-dir results/my-workflows-cpu
# Repetir después con --device mps y otra carpeta, si está disponible.
```

La demo local permite elegir escenario, editar sus entradas y descargar la respuesta. [CPU](results/workflows-cpu/summary.json), [GPU](results/workflows-mps/summary.json) y [offline](results/workflows-offline/summary.json) incluyen métricas; cada carpeta contiene también las respuestas completas, errores, Brier y matrices de confusión. [Verificación](results/workflows-verification.json).

Para comprobar la evidencia publicada: `.venv/bin/python verify_workflows.py --output results/my-verification.json`. Para verificar repeticiones propias, indicar también sus carpetas con `--cpu-dir`, `--mps-dir` y `--offline-dir`.

![Falso positivo real: el modelo bloquea una cita cuya etiqueta es permitir](docs/workflows.png)
