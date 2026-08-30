## Business Case — Predicción de ocurrencia de incendios forestales

Fase 1 de la guía orientativa. Documenta el problema, la justificación del enfoque de ML y los criterios de éxito antes de entrar en modelado, y sirve de base para las primeras celdas del notebook final.

### 1. Contexto y problema de negocio

La gestión de la prevención y respuesta ante incendios forestales requiere anticipar, con la mayor antelación posible, en qué condiciones meteorológicas y geográficas es más probable que se produzca un incendio. Hoy esa valoración se apoya en índices meteorológicos agregados (como el FWI) y en la experiencia de los equipos de vigilancia, sin un modelo que combine de forma sistemática el conjunto de variables ambientales disponibles.

Un modelo que prediga la probabilidad de ocurrencia de incendio a partir de condiciones meteorológicas permite:

- Priorizar la vigilancia y los recursos de prevención en las zonas y momentos de mayor riesgo real.
- Reducir el número de incendios no detectados a tiempo (falsos negativos), cuyo coste —humano, ecológico y económico— es muy superior al de una alerta que finalmente no se materializa.
- Evitar, dentro de lo posible, la sobrealerta sistemática, que satura los recursos disponibles y erosiona la confianza en el sistema.

### 2. Hipótesis y objetivo de modelado

- **Tipo de problema**: clasificación binaria.
- **Target**: `occured` — indica si se detectó (1) o no (0) un incendio en el punto y periodo de observación.
- **Hipótesis de partida**:
  - Las condiciones meteorológicas de corto plazo (temperatura, humedad, viento, radiación solar, punto de rocío) junto con el índice compuesto FWI son suficientes para discriminar, en un grado razonable, entre situaciones de riesgo alto y bajo.
  - El coste de un falso negativo (no detectar un incendio real) es sustancialmente mayor que el de un falso positivo, lo que condiciona la métrica de evaluación prioritaria.
  - Existe cierto desequilibrio en variables como `daynight_N`, cuyo efecto sobre el modelo conviene monitorizar sin asumir de partida que requiera corrección.

### 3. Criterios de éxito

El modelo no sustituye el criterio de los equipos de extinción y prevención — lo complementa con una estimación objetiva y verificable del riesgo, priorizada según el coste asimétrico de los errores:

- **Recall de la clase positiva ≥ 70%**: el modelo debe detectar correctamente al menos 7 de cada 10 incendios reales, minimizando los falsos negativos. Este umbral se fija como objetivo exigente pero realista, y no más ambicioso (por ejemplo, 90%), por dos motivos. Primero, un recall del 100% no resulta plausible en este dominio: la ocurrencia de un incendio depende de factores no capturados por las variables meteorológicas disponibles (actividad humana, negligencias, causas antrópicas intencionadas, variabilidad microclimática local no representada en datos agregados), lo que impone un techo de rendimiento inherente al problema con el conjunto de variables predictoras disponible. Segundo, exigir un recall superior implica, en la práctica, sacrificar de forma severa el recall de la clase negativa —tal y como se observó empíricamente durante la fase de optimización, donde configuraciones orientadas a maximizar el recall por encima del 80% derivaron en un modelo que clasificaba la mayoría de los casos como positivos (recall de clase negativa ≈ 42%), resultando operativamente inviable por el volumen de falsas alertas generadas.
- **Recall de la clase negativa > 50%**: el modelo debe mantener una capacidad mínima de discriminación real entre clases, evitando el escenario en que prácticamente todo se clasifique como riesgo de incendio, lo cual saturaría el sistema de alerta y lo haría operativamente inútil.
- Ambos umbrales se combinan con **F1-score** como métrica de equilibrio durante la fase de optimización de hiperparámetros.

### 4. Plan de acción

El modelo alimenta una decisión operativa recurrente (asignación de vigilancia y recursos de prevención según el riesgo estimado), no un informe puntual. Se evalúan varias familias de algoritmos (ensembles de árboles y modelos basados en distancias) y, dada la similitud de rendimiento observada entre ellos en el baseline, se explora un ensemble por votación que combine ambos enfoques.

Fuera de alcance de este proyecto (posible extensión futura) quedarían la incorporación de variables ambientales adicionales (precipitación, materia orgánica del suelo, tipo de vegetación) y validación con esquema de series temporales. Se identifican como líneas de mejora en las conclusiones, pero no forman parte del alcance actual.

### 5. Requerimientos de los datos y origen del dataset

El dataset (`final_dataset.csv`) parte del *Global Wildfire Dataset* publicado en Kaggle por Vijayaragul VR ([kaggle.com/datasets/vijayaragulvr/wildfire-prediction](https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction)), y cumple con los requerimientos definidos para abordar el problema:

- Registro histórico de observaciones meteorológicas por punto geográfico, con indicador de ocurrencia de incendio.
- Variables meteorológicas horarias/diarias (temperatura, humedad, viento, presión, radiación, punto de rocío, cobertura nubosa, evapotranspiración) y el índice compuesto FWI.
- Coordenadas geográficas (latitud/longitud) e indicador día/noche.

### 6. Disponibilidad

Dataset (`final_dataset.csv`) construido a partir de variables meteorológicas obtenidas de Open-Meteo, combinadas con los registros de ocurrencia de incendio y el FRP (Fire Radiative Power) como variable adicional orientada a tareas de regresión, fuera del alcance de este proyecto de clasificación.

### 7. Calidad

- Target binario (`occured`) con una distribución de clases razonablemente equilibrada (confirmado en el EDA y reflejado en el `support` de los `classification_report` obtenidos: ~11.979 vs ~11.793 casos en el conjunto de test).
- Presencia de variables con correlación de Spearman elevada (> 0.8) entre sí, tratadas en la fase de preprocesado para evitar redundancia, especialmente relevante para los modelos lineales.
- Variables numéricas con distribuciones sesgadas, corregidas mediante transformación de potencia (Yeo-Johnson) antes de su uso en modelos sensibles a la escala (regresión logística, KNN).

Referencias: notebooks de EDA dirigido, feature engineering, preprocesado, modelado y optimización del propio proyecto.



