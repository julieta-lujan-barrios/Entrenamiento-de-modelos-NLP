# Sentiment Analysis — Taller II

Sistema de análisis de sentimiento desarrollado para la asignatura **Taller II** (Ingeniería en Inteligencia Artificial, UNSTA), en el marco de un caso propuesto por la empresa ficticia **Strata Analytics**. El objetivo fue construir un pipeline completo de PLN capaz de clasificar reseñas de clientes como **positivas**, **negativas** o **neutrales**.

> 📄 Informe completo: [`INFORME_FINAL_-_TALLER_II.pdf`](INFORME_FINAL_-_TALLER_II.pdf)

## Equipo

- Barrios, Julieta Luján
- Chalin, Matías Alejandro
- Granito, Leandro Elio
- Sappia, Lucio Agustín

**Docente a cargo:** Ana Martínez Saucedo

## Descripción del proyecto

El desafío consistió en recorrer el ciclo de vida completo de una solución de PLN aplicada a reseñas reales:

1. **Recolección de datos** mediante *web scraping* de reseñas de Airbnb publicadas en [Trustpilot](https://www.trustpilot.com/review/www.airbnb.com), obteniendo un corpus inicial de **12.587 comentarios**.
2. **Procesamiento y limpieza** del texto (normalización a inglés, expansión de abreviaciones, eliminación de stopwords/puntuación, *stemming* y *lemmatization*).
3. **Balanceo del dataset**, dado el severo sesgo hacia reseñas negativas (88,7% negativas vs. 9,5% positivas vs. 1,8% neutrales). Se generaron dos datasets experimentales por *oversampling*, quedándose finalmente con uno de **33.513 reseñas** balanceadas en partes iguales.
4. **Vectorización de texto** con Bag of Words (BoW) y TF-IDF.
5. **Entrenamiento y comparación de 6 modelos**: Naive Bayes, Regresión Logística, SVM (LinearSVC), Random Forest, MLP (red neuronal) y XGBoost.
6. **Optimización de hiperparámetros** del modelo ganador mediante GridSearchCV.
7. **Inferencia y despliegue** con una interfaz interactiva en Gradio.

## Resultado final

El mejor modelo fue una **Red Neuronal Multicapa (MLP)** con vectorización **TF-IDF**, optimizada con `GridSearchCV` (arquitectura de 1 capa oculta de 64 neuronas, activación `relu`, `learning_rate_init=0.01`).

| Métrica | Pre-optimización | Post-optimización |
|---|---|---|
| Accuracy | 0.979 | **0.985** |
| Macro-F1 | 0.980 | **0.985** |
| Weighted-F1 | 0.980 | **0.985** |

## Limitaciones y aprendizajes

El dataset balanceado numéricamente (33.513 reseñas, 11.171 por clase) no está balanceado en **diversidad léxica**: las 11.171 reseñas negativas son todas únicas, mientras que las clases positiva y neutral se obtuvieron duplicando por *oversampling* solo 1.191 y 225 reseñas originales respectivamente. Esto genera una leve tendencia del modelo a clasificar como negativas frases neutrales o ambiguas. Una mejora futura sería aplicar *data augmentation* con IA generativa para crear ejemplos sintéticos en vez de duplicar los existentes.

## Estructura del repo

El pipeline se organiza en scripts secuenciales, uno por etapa del proceso:

```
├── docs/
│   └── INFORME_FINAL_-_TALLER_II.pdf
├── src/
│   ├── 01_creacion_dataset_inicial.py            # web scraping y armado del dataset base
│   ├── 02_dataset_preprocesado.py                # limpieza, lematización, stemming
│   ├── 03_dataset_balanceado.py                  # balanceo por oversampling
│   ├── 04_entrenamiento_y_eleccion_de_modelo.py  # comparación de los 6 modelos
│   ├── 05_optimizacion_del_mejor_modelo.py       # GridSearchCV sobre el MLP
│   └── 06_inferencia_final.py                    # predicción + interfaz Gradio
└── README.md
```

## Tecnologías utilizadas

- Python (scikit-learn, XGBoost, NLTK/spaCy, Matplotlib)
- Gradio (interfaz de inferencia)
- Web scraping (Trustpilot)

## Referencias

Ver bibliografía completa en el [informe final](INFORME_FINAL_-_TALLER_II.pdf).
