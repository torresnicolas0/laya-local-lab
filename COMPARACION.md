# Tres variantes, los mismos diez ejemplos — 23/09/2026

**Elegir variante cambió mucho el resultado:** Typed-Decisions acertó los diez departamentos y las diez decisiones de devolución. Laya base tuvo el menor error de urgencia; Multilingual fue el más rápido.

## Método

Mismo M1 Max de 32 GB, Python 3.12.8, Laya 0.3.11 y dependencias de la prueba inicial. Los tres checkpoints pertenecen a la revisión fijada en `models.lock.json`. Laya base y Typed-Decisions usan ModernBERT-large; Multilingual usa mmBERT-base. Sus contextos son 512, 1024 y 1024 tokens respectivamente; todos los textos caben completos.

Se reutilizaron **las diez entradas inglesas**, preguntas y etiquetas originales, sin modificar ni entrenar. Inglés es el ámbito común de las tres variantes. Los 30 textos ES/PT/EN siguen siendo una evaluación separada de Multilingual.

Cada modelo corre en un proceso nuevo y en orden fijo: Multilingual, base, Typed-Decisions. CPU y GPU se ejecutan por separado, con cuatro hilos, tres calentamientos y una pasada por texto. Cada tiempo incluye tres preguntas; la GPU se sincroniza. No es una prueba de carga ni un benchmark estadístico.

## Calidad

| Variante | Departamento | Devolución, umbral 0,5 | Urgencia MAE, 0–2 | Brier devolución |
|---|---:|---:|---:|---:|
| Multilingual | 6/10 | 6/10 | 0,814 | 0,204 |
| Laya base | 9/10 | 9/10 | **0,131** | 0,059 |
| Typed-Decisions | **10/10** | **10/10** | 0,263 | **0,038** |

La categoría mayoritaria daría 3/10; responder siempre «no hay devolución» daría 8/10. CPU y GPU obtuvieron los mismos aciertos.

En «I do not want a refund», la probabilidad de devolución bajó del 81,2 % de Multilingual al 29,9 % de base y al 15,7 % de Typed-Decisions. Ante una factura ya resuelta seguida de una solicitud comercial, solo Typed-Decisions eligió ventas. Base conservó un falso positivo de devolución en la caída del checkout.

## Velocidad y memoria

| Variante | Mediana CPU | Mediana GPU | Pico RSS CPU |
|---|---:|---:|---:|
| Multilingual | 125,0 ms | 35,0 ms | 2.200 MiB |
| Laya base | 267,1 ms | 65,4 ms | 2.838 MiB |
| Typed-Decisions | 280,4 ms | 66,3 ms | 2.830 MiB |

RSS es memoria del proceso; no mide toda la memoria unificada/GPU. Los JSON incluyen p95, carga, memoria y respuestas. La repetición en otro entorno, con IPv4/IPv6 bloqueados por macOS, produjo respuestas idénticas a CPU para las tres variantes.

## Interpretación

El 10/10 corresponde a estos diez ejemplos sintéticos conocidos, **no a una garantía de fiabilidad**. No hubo conjunto independiente de validación ni evaluación de las variantes inglesas en español. No se puede atribuir toda la diferencia al entrenamiento: Multilingual también tiene otra arquitectura.

El SDK avisó de un ajuste de temperatura para preguntas con 11 o más opciones en los checkpoints ingleses; nuestra clasificación usa cuatro opciones. No se realizó calibración adicional. La [ficha oficial](https://huggingface.co/convaiinnovations/laya-typed-decisions) también documenta límites de calibración.

Evidencia: [CPU](results/comparison-cpu/summary.json), [GPU](results/comparison-mps/summary.json), [repetición offline](results/comparison-offline/summary.json). Cada carpeta conserva las respuestas completas. La [prueba inicial](RESULTADOS.md) permanece intacta.
