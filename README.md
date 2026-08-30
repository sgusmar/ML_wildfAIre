# ML_wildfAIre — Predicción de ocurrencia de incendios forestales

Proyecto de Machine Learning sobre la ocurrencia de incendios forestales a partir de condiciones meteorológicas: clasificar si, dado un conjunto de variables ambientales, se produce (o no) un incendio, como apoyo a la priorización de vigilancia y recursos de prevención.

![Wildfire risk banner](src\img\logo_fire.png)

[Español](#español) / [English](#english)

## Español

## Índice

- [Planteamiento del problema](#planteamiento-del-problema)
- [Datos](#datos)
- [Enfoque y pipeline](#enfoque-y-pipeline)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Stack técnico](#stack-técnico)
- [Cómo reproducirlo](#cómo-reproducirlo)
- [Resultados](#resultados)
- [Limitaciones y próximos pasos](#limitaciones-y-próximos-pasos)

## Planteamiento del problema

La vigilancia y prevención de incendios forestales se apoya hoy en índices meteorológicos agregados (como el FWI) y en la experiencia de los equipos sobre el terreno, sin un modelo que combine de forma sistemática el conjunto de variables ambientales disponibles.

**Objetivo**: predecir la probabilidad de ocurrencia de un incendio (`occured`, target binario) a partir de variables meteorológicas y del propio FWI, priorizando no dejar pasar incendios reales sin, por ello, generar un volumen de falsas alertas que sature los recursos disponibles.

El razonamiento completo del caso de negocio, las hipótesis de partida y los criterios de éxito están documentados en `src/docs/business_case.md`.

## Datos

Dataset (`final_dataset.csv`) construido a partir del *Global Wildfire Dataset* publicado en Kaggle por Vijayaragul VR ([kaggle.com/datasets/vijayaragulvr/wildfire-prediction](https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction)), enriquecido con variables meteorológicas horarias de [Open-Meteo](https://open-meteo.com/) y el índice compuesto Fire Weather Index (FWI).

- **Target**: `occured` (1 = incendio detectado, 0 = no detectado), con clases razonablemente equilibradas.
- **Variables**: temperatura, humedad, viento (velocidad y dirección), presión, radiación solar, punto de rocío, cobertura nubosa, evapotranspiración, coordenadas geográficas, indicador día/noche (`daynight_N`) y FWI.
- Variable adicional `frp` (Fire Radiative Power), no utilizada en este proyecto de clasificación — orientada a tareas de regresión.

## Enfoque y pipeline

Clasificación binaria supervisada, desarrollada en fases independientes y ensambladas en un notebook final (`main_definitive.ipynb`):

1. **EDA dirigido** (`EDA.ipynb`) — distribución de variables, correlación de Spearman entre features (eliminación de pares con |ρ| > 0,8 para evitar redundancia, especialmente relevante en modelos lineales), y revisión del desequilibrio en `daynight_N`.
2. **Feature engineering** (`feature_engineering.ipynb`) — cálculo del Déficit de Presión de Vapor (VPD) a partir del punto de rocío, entre otras variables derivadas.
3. **Preprocesado** (`preprocessing.ipynb`) — transformación de potencia (Yeo-Johnson) para corregir la asimetría de las variables numéricas antes de su uso en modelos sensibles a la escala (regresión logística, KNN).
4. **Modelado** (`model_optimization_evaluation.ipynb`) — comparativa de baseline entre modelos basados en ensembles de árboles (Random Forest, XGBoost, LightGBM, GradientBoosting) y modelos basados en distancias (Regresión Logística, KNN), con validación cruzada (K-Fold).
5. **Optimización** — búsqueda de hiperparámetros en dos etapas (búsqueda amplia + refinamiento acotado) por modelo, con recall como métrica inicial y F1 como métrica de equilibrio tras detectar un desequilibrio severo entre clases en configuraciones muy agresivas.
6. **Ensemble** — Voting Classifier combinando modelos de ambas familias (árboles + distancias), dada la similitud de rendimiento individual observada en el baseline y la baja correlación esperada entre sus errores.
7. **Evaluación final** — una única evaluación contra el conjunto de test, con matriz de confusión, curva ROC-AUC e importancia de features (`gain`).
8. **Persistencia** — modelo y scaler finales guardados en `src/models/` (`joblib`).

## Estructura del repositorio
├── src/
│ ├── data_sample/ # Dataset y splits train/test
│ ├── docs/ # business_case.md
│ ├── models/ # Modelo y scaler entrenados (joblib)
│ ├── notebooks/ # Un notebook por fase del pipeline
│ │ ├── data_understanding.ipynb
│ │ ├── EDA.ipynb
│ │ ├── feature_engineering.ipynb
│ │ ├── preprocessing.ipynb
│ │ └── model_optimization_evaluation.ipynb
│ └── utils/ # Funciones reutilizadas (data_utils.py)
├── main_definitive.ipynb # Pipeline completo ensamblado
├── requirements.txt
└── README.md


## Stack técnico

- **Python** (pandas, numpy, scipy)
- **scikit-learn** — modelado, validación cruzada, búsqueda de hiperparámetros, `PowerTransformer`
- **LightGBM, XGBoost** — modelos de ensemble por gradient boosting
- **matplotlib, seaborn** — visualización
- **joblib** — persistencia del modelo y del scaler
- **Jupyter Notebook**
- **Git / GitHub** — ramas `main`/`dev`, Pull Requests

## Cómo reproducirlo

```bash
git clone https://github.com/sgusmar/ML_wildfAIre.git
cd ML_wildfAIre
pip install -r requirements.txt
```

Para usar directamente el modelo ya entrenado, sin reejecutar todo el pipeline:

```python
import joblib

modelo = joblib.load("src/models/model_final.joblib")
scaler = joblib.load("src/models/scaler.joblib")

# modelo.predict(X) sobre un DataFrame con las mismas columnas del entrenamiento
```

## Resultados

Métricas principales: **recall** (clase positiva, prioritaria por el coste asimétrico de los falsos negativos) y **F1-score** como equilibrio con la clase negativa.

| Modelo | Recall (clase 1) | Precisión | F1 | Accuracy |
|---|---|---|---|---|
| Random Forest (baseline) | 0,781 | 0,603 | 0,681 | 0,633 |
| XGBoost (baseline) | 0,765 | 0,617 | 0,684 | 0,645 |
| LightGBM (baseline) | 0,766 | 0,626 | 0,689 | 0,654 |
| LightGBM (optimizado) | 0,769 | 0,630 | 0,693 | 0,658 |
| **Voting Ensemble (final)** | **0,776** | **0,629** | **0,695** | **0,658** |

Sobre el conjunto de test, el modelo final detecta correctamente ~4 de cada 5 incendios reales (recall clase 1: 0,78) y clasifica correctamente más del 50% de los casos negativos (recall clase 0: 0,54) — cumpliendo ambos criterios de éxito fijados en el caso de negocio (recall positivo ≥ 70%, recall negativo > 50%).

El ensemble por votación mejora ligeramente el área bajo la curva ROC respecto al mejor modelo individual (LightGBM), aunque las métricas puntuales a umbral 0,5 resultan muy similares entre ambos — el valor añadido del ensemble es marginal en este proyecto.

## Limitaciones y próximos pasos

- Desequilibrio en la representación de `daynight_N`, asociado a una distribución bimodal observada en las probabilidades predichas por LightGBM; ampliar la recogida de casos diurnos podría mejorar la calibración.
- Incorporar variables ambientales adicionales (precipitación, materia orgánica del suelo, tipo de vegetación) para capturar interacciones actualmente no representadas.
- Explorar estrategias de combinación más sofisticadas para el ensemble (stacking, ponderación optimizada).
- Validar la estabilidad temporal del modelo con un esquema de validación cronológico, en lugar de validación cruzada aleatoria.

---

## English

### Problem description

Wildfire monitoring and prevention currently relies on aggregated weather indices (such as the FWI) and field-team experience, without a model that systematically combines the full set of available environmental variables.

**Goal**: predict the probability of wildfire occurrence (`occured`, binary target) from weather variables and the FWI index, prioritizing the detection of real wildfires without generating a volume of false alerts that would overwhelm available prevention resources.

Full business case, modeling hypothesis and success criteria: [`src/docs/business_case.md`](src/docs/business_case.md).

### Dataset

Built from the [*Global Wildfire Dataset*](https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction) published on Kaggle by Vijayaragul VR, enriched with hourly weather variables from [Open-Meteo](https://open-meteo.com/) and the composite Fire Weather Index (FWI).

- **Target**: `occured` (1 = wildfire detected, 0 = not detected), reasonably balanced classes.
- **Features**: temperature, humidity, wind (speed and direction), pressure, solar radiation, dewpoint, cloud cover, evapotranspiration, geographic coordinates, day/night indicator (`daynight_N`) and FWI.
- Additional `frp` (Fire Radiative Power) column, not used in this classification project — intended for regression tasks.

### Solution

Supervised binary classification, developed in independent phases and assembled into a final notebook (`main_definitive.ipynb`):

Pipeline: directed EDA (feature distributions, Spearman correlation filtering above |ρ| > 0.8) → feature engineering (Vapor Pressure Deficit from dewpoint, among others) → preprocessing (Yeo-Johnson power transform for scale-sensitive models) → baseline comparison across tree-based ensembles (Random Forest, XGBoost, LightGBM, GradientBoosting) and distance-based models (Logistic Regression, KNN) → two-stage hyperparameter search (recall, then F1 after detecting severe class imbalance under overly aggressive configurations) → Voting Classifier ensemble combining both model families → single final evaluation against the test set (confusion matrix, ROC-AUC, gain-based feature importance) → persistence of the final model and scaler in `src/models/` (`joblib`).

### Repository structure

See the Spanish section above — same structure.

### Tech stack

Python ([pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [scipy](https://scipy.org/)), [scikit-learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/), [XGBoost](https://xgboost.readthedocs.io/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [joblib](https://joblib.readthedocs.io/), Jupyter Notebook, Git/GitHub.

### Reproduction

```bash
git clone https://github.com/sgusmar/ML_wildfAIre.git
cd ML_wildfAIre
pip install -r requirements.txt
```

To use the already-trained model directly, without re-running the full pipeline, load `src/models/model_final.joblib` and `src/models/scaler.joblib` with `joblib` (see the Spanish section for the exact snippet).

### Main results

Final model: **Voting Ensemble** (LightGBM, XGBoost, Logistic Regression, KNN), evaluated once against the untouched test set:

| Model | Recall (class 1) | Precision | F1 | Accuracy |
|---|---|---|---|---|
| Random Forest (baseline) | 0.781 | 0.603 | 0.681 | 0.633 |
| XGBoost (baseline) | 0.765 | 0.617 | 0.684 | 0.645 |
| LightGBM (baseline) | 0.766 | 0.626 | 0.689 | 0.654 |
| LightGBM (optimized) | 0.769 | 0.630 | 0.693 | 0.658 |
| **Voting Ensemble (final)** | **0.776** | **0.629** | **0.695** | **0.658** |

The final model correctly detects ~4 out of 5 real wildfires (class 1 recall: 0.78) while correctly classifying over 50% of negative cases (class 0 recall: 0.54), meeting both success criteria defined in the business case. The voting ensemble slightly improves ROC-AUC over the best individual model (LightGBM), though point metrics at the default threshold are very similar between both — the added value of the ensemble is marginal in this project. See the Spanish section for the full limitations and improvement roadmap.

### Author

- sgusmar — [github.com/sgusmar](https://github.com/sgusmar)