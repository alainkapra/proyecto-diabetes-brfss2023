# Predicción de diabetes y segmentación de perfiles de riesgo · BRFSS 2023

**Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial**: Maestría en Inteligencia Artificial, Universidad de La Salle (Bogotá), 2026.
Autor: Alain Alexander Camacho Prado. Asignatura: Ciencia de Datos. Docente: Fabián Camilo Castro Riveros.

> **Aplicación en vivo:** [alainkapra-diabetes-brfss2023.streamlit.app](https://alainkapra-diabetes-brfss2023.streamlit.app) · Despliegue documentado en [docs/DESPLIEGUE.md](docs/DESPLIEGUE.md).

Solución de ciencia de datos sobre **272.769 adultos** de la encuesta *Behavioral Risk Factor Surveillance System* (BRFSS) 2023 de los CDC. Cubre cinco frentes:

- **Clasificación** del diagnóstico de diabetes con regresión logística, Random Forest, XGBoost y un perceptrón multicapa en PyTorch.
- **Regresión** del índice de masa corporal con Ridge, Lasso, ElasticNet, Random Forest y XGBoost.
- **Segmentación** de perfiles de riesgo con K-Means, DBSCAN y el método jerárquico de Ward.
- **Interpretabilidad** con valores SHAP.
- **Aplicación web interactiva** desplegada en Streamlit.

## Resultados principales

Las métricas se calcularon sobre el conjunto de prueba (n = 40.916).

| Indicador | Valor |
|---|---|
| AUC-ROC del clasificador (XGBoost) | 0,816 (IC 95 %: 0,810–0,821) |
| AUC-PR (línea base = 0,146) | 0,413 |
| Sensibilidad de tamizaje (umbral 0,5) | 78,7 % |
| Valor predictivo positivo (umbral 0,5) | 30,6 % |
| F1 de la clase diabetes (umbral óptimo 0,62) | 0,462 |
| R² del IMC (XGBoost) | 0,186 |
| Perfiles de riesgo (K-Means, k = 3) | prevalencias de 3,7 %, 22,1 % y 27,7 % |

## Estructura del repositorio

```
proyecto-diabetes-brfss2023/
├── data/
│   ├── raw/                     CSV original (cdc_brfss_diabetes_2023.csv)
│   └── processed/               datos con clúster y partición, muestra PCA y predicciones de prueba (parquet)
├── notebooks/
│   ├── 01_Fase2_Preprocesamiento_Baseline.ipynb
│   └── 02_Fase3_Modelado_Clustering_Optimizacion.ipynb
├── src/
│   ├── preprocesamiento.py      transformadores, variables derivadas y pipelines sin fuga de datos
│   └── entrenar_modelos.py      reentrena los modelos finales y genera los artefactos de la aplicación
├── dashboard/
│   ├── app.py                   punto de entrada de Streamlit (filtros y navegación)
│   ├── utils.py                 carga en caché, filtros, sistema visual y componentes animados
│   ├── vistas/                  panorama, explorador, predicción, segmentación, desempeño, método
│   ├── static/fonts/            tipografías Sora y Manrope (licencia SIL OFL)
│   └── requirements.txt         dependencias mínimas para el despliegue
├── models/                      modelos finales (.joblib) y metadatos (métricas, umbral, versiones)
├── reports/
│   ├── FASE_4_Articulo_IEEE_...pdf   informe técnico final (artículo IEEE)
│   ├── articulo_latex/          fuentes LaTeX del artículo (main.tex, referencias.bib, tablas, figuras)
│   ├── FASE_1 … FASE_3 (PDF)    informes de las fases anteriores
│   └── resultados/, figuras/, artefactos_fase2/, artefactos_fase3/
├── docs/DESPLIEGUE.md           guía de despliegue en Streamlit Community Cloud
├── .streamlit/config.toml       tema oscuro de la aplicación
├── requirements.txt             entorno completo con versiones exactas (pip)
└── environment.yml              el mismo entorno para conda
```

## Ejecución inmediata

Requiere Python 3.13 (se probó con la versión 3.13.18).

```bash
git clone https://github.com/alainkapra/proyecto-diabetes-brfss2023.git
cd proyecto-diabetes-brfss2023

# opción A: pip
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# opción B: conda
conda env create -f environment.yml && conda activate diabetes-brfss2023

python -m src.entrenar_modelos      # opcional: reentrena (≈ 15 s) y reproduce las métricas de prueba
streamlit run dashboard/app.py      # abre la aplicación en http://localhost:8501
```

Para reproducir las fases 2 y 3, ejecute en orden los cuadernos de `notebooks/`. La fase 3 tarda unos 8 minutos en CPU.

El artículo se compila con `cd reports/articulo_latex && python generar_tablas.py && latexmk -pdf main.tex`. El script `generar_tablas.py` toma las tablas directamente de los resultados en JSON.

## Módulos de la aplicación

La guía pide cuatro módulos obligatorios:

| Módulo de la guía | Dónde está | Interacción |
|---|---|---|
| **Filtros dinámicos** | Panel lateral (todas las páginas) | Año de la encuesta, rangos de edad e IMC, sexo, salud, ingreso, educación, condiciones, perfil y diagnóstico. En el Panorama, la selección de un rango de edades filtra los otros gráficos |
| **Tarjetas KPI ejecutivas** | Panorama y Desempeño del modelo | Contadores animados de la población y del rendimiento (AUC-ROC, AUC-PR, sensibilidad, VPP, F1, R², silueta) |
| **Predicción en vivo** | Predicción en vivo | Formulario del perfil, medidor de riesgo, ubicación en la distribución, cascada SHAP, perfil asignado y escenarios «¿qué pasaría si…?» |
| **Segmentación (clustering)** | Segmentación | Dispersión 3D giratoria y 2D en el espacio PCA (por perfil o por diagnóstico), radar, mapa y barras de los perfiles promedio, y validación de k |

Además de esos cuatro módulos, la aplicación incluye:

- un gráfico animado de prevalencia por edad;
- un sunburst que se explora con clics;
- un diagrama de categorías paralelas con las rutas hacia la diabetes;
- el umbral de decisión interactivo, con un retrato de 100 personas tamizadas;
- la reproducción, época por época, del entrenamiento de la red neuronal.

## Datos y licencia

- Fuentes:
  - Centers for Disease Control and Prevention (2023), *BRFSS survey data*.
  - Van Thieu (2024), *Diabetes Health Indicators Dataset (BRFSS 2015–2023)*, publicado en Kaggle.
- Es un trabajo académico: **no es una herramienta de diagnóstico** ni sustituye la valoración médica.
