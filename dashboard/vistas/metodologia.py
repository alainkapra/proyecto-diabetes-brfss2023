"""Método: pasos del proyecto, arquitectura sin fuga de datos, herramientas, diccionario con tipos de variables,
limitaciones y reproducibilidad."""
import streamlit as st

from utils import CIAN, MAGENTA, VIOLETA, cargar_modelos, hero, nota, panel, seccion

PASOS = [
    ("Datos", "BRFSS 2023 de los CDC: 272.769 adultos y 18 variables autorreportadas, sin faltantes."),
    ("Limpieza", "Duplicados conservados (perfiles reales); IMC winsorizado a p1–p99 aprendidos en entrenamiento."),
    ("Ingeniería", "CardioRisk, categoría de IMC, días de malestar e interacción edad × IMC dentro de un Pipeline."),
    ("Partición", "70/15/15 estratificada con semilla fija. La prueba queda sellada hasta el final."),
    ("Modelado", "Logística, Random Forest, XGBoost y red neuronal en PyTorch; Ridge, Lasso, ElasticNet y ensambles para el IMC."),
    ("Optimización", "RandomizedSearchCV con validación cruzada estratificada; AUC-PR por el desbalance 1 a 5,8."),
    ("Segmentación", "K-Means por codo y silueta; contraste con DBSCAN y Ward; PCA para visualizar."),
    ("Despliegue", "Esta app en Streamlit con los mismos modelos evaluados en prueba y explicaciones SHAP."),
]

DOT = f"""
digraph G {{
  rankdir=TB; bgcolor="transparent"; nodesep=.35; ranksep=.38;
  node [shape=box, style="rounded,filled", fontname="Manrope", fontsize=13, color="#3a4380", fillcolor="#141a3d", fontcolor="#e7ecff", margin="0.2,0.1", penwidth=1.2];
  edge [color="#6b7699", arrowsize=.6, penwidth=1];
  csv [label="CSV BRFSS 2023\\n272.769 × 18"];
  split [label="Partición estratificada\\n70 / 15 / 15", color="{CIAN}", penwidth=2];
  tr [label="Entrenamiento\\n190.938"]; va [label="Validación\\n40.915"]; te [label="Prueba sellada\\n40.916", color="{MAGENTA}", fontcolor="{MAGENTA}"];
  pipe [label="Pipeline sin fuga\\nrecorte · derivadas · imputación\\ncodificación · escala"];
  clf [label="Clasificación\\nlogística · RF · XGBoost · MLP"]; reg [label="Regresión IMC\\nRidge · Lasso · EN · RF · XGB"];
  clu [label="Segmentación\\nK-Means · Ward · DBSCAN"];
  cv [label="Hiperparámetros\\nCV estratificada (AUC-PR)"]; thr [label="Umbral óptimo\\nen validación"];
  ev [label="Evaluación final\\nmétricas · bootstrap · SHAP", fillcolor="#3b1d5e", color="{VIOLETA}"];
  app [label="App Streamlit\\ndesplegada", fillcolor="{CIAN}", fontcolor="#0a0f26", color="{CIAN}"];
  {{rank=same; tr; va; te;}} {{rank=same; clf; reg; clu;}}
  csv -> split; split -> tr; split -> va; split -> te; tr -> pipe; pipe -> clf; pipe -> reg; pipe -> clu; clf -> cv; cv -> thr; va -> thr;
  thr -> ev; te -> ev; reg -> ev; clu -> ev; ev -> app;
}}
"""

STACK = [("Python", "3.13"), ("pandas", "3.0.5"), ("NumPy", "2.5.3"), ("scikit-learn", "1.9.1"), ("imbalanced-learn", "0.14.2"), ("XGBoost", "3.4.1"),
         ("PyTorch", "2.15.0"), ("SHAP", "0.54.1"), ("SciPy", "1.18.1"), ("Plotly", "7.1.0"), ("Streamlit", "1.67.0"), ("joblib", "1.6.0"),
         ("pyarrow", "26.0.1"), ("matplotlib", "3.11.2"), ("seaborn", "0.13.2"), ("nbclient", "0.11.0")]

DICC = [("Diabetes_binary", "Nominal binaria", "Objetivo (clasificación)", "0 = no, 1 = prediabetes o diabetes"),
        ("BMI", "Cuantitativa continua (razón)", "Objetivo (regresión) y predictor", "12 a 100 kg/m²"),
        ("HighBP, HighChol, CholCheck", "Nominal binaria", "Predictor", "0 = no, 1 = sí"),
        ("Smoker, Stroke, HeartDiseaseorAttack", "Nominal binaria", "Predictor", "0 = no, 1 = sí"),
        ("PhysActivity, NoDocbcCost, DiffWalk", "Nominal binaria", "Predictor", "0 = no, 1 = sí"),
        ("Sex", "Nominal binaria", "Predictor", "0 = femenino, 1 = masculino"),
        ("GenHlth", "Ordinal", "Predictor", "1 = excelente … 5 = mala"),
        ("Age", "Ordinal", "Predictor", "1 = 18–24 … 13 = 80 o más"),
        ("Education", "Ordinal", "Predictor", "códigos 2, 4, 5 y 6"),
        ("Income", "Ordinal", "Predictor", "códigos 1 a 7"),
        ("MentHlth, PhysHlth", "Cuantitativa discreta (conteo)", "Predictor", "0 a 30 días"),
        ("CardioRisk", "Derivada (conteo)", "Predictor (árboles)", "suma de 4 condiciones, 0 a 4"),
        ("UnhealthyDays, Age_BMI", "Derivadas cuantitativas", "Predictor", "min(30, días); edad × IMC")]


def vista():
    _, _, _, meta = cargar_modelos()
    hero("Método y reproducibilidad", "Cómo se construyó la solución",
         "Del archivo de la encuesta a la app desplegada: el flujo de ciencia de datos, las decisiones que evitan la fuga de datos, las herramientas "
         "con sus versiones exactas y la manera de reproducir cada resultado desde el repositorio.")
    for j, fila in enumerate((PASOS[:4], PASOS[4:])):
        cols = st.columns(4, gap="small")
        for i, ((t, d), c) in enumerate(zip(fila, cols)):
            c.markdown(f'<div class="paso"><span class="n">{4 * j + i + 1:02d}</span><b>{t}</b><span>{d}</span></div>', unsafe_allow_html=True)
        st.write("")
    seccion("Arquitectura del flujo", "La partición ocurre antes de cualquier transformación; todo lo que se aprende de los datos se aprende solo con entrenamiento.")
    with panel("dot"):
        st.graphviz_chart(DOT, width="stretch")
    nota("Percentiles de recorte, medianas, codificaciones, escalador, K-Means y PCA se ajustan con entrenamiento; el umbral se elige en validación y el conjunto "
         "de prueba se usa una sola vez. El mismo código vive en <b>src/preprocesamiento.py</b>, que comparten el script de entrenamiento y esta app.")
    seccion("Herramientas")
    st.markdown('<div class="stack">' + "".join(f"<span>{n}<b>{v}</b></span>" for n, v in STACK) + "</div>", unsafe_allow_html=True)
    seccion("Diccionario de variables")
    filas = "".join(f'<tr><td style="color:#fff">{v}</td><td>{t}</td><td>{r}</td><td>{c}</td></tr>' for v, t, r, c in DICC)
    st.markdown(f'<table class="tabla-ed"><thead><tr><th>Variable</th><th>Tipo</th><th>Rol</th><th>Codificación</th></tr></thead><tbody>{filas}</tbody></table>',
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a:
        seccion("Limitaciones")
        nota("<b>Autorreporte:</b> peso, enfermedades y hábitos dependen de la memoria de cada persona.<br><b>Diabetes diagnosticada:</b> los casos no "
             "diagnosticados figuran como sanos.<br><b>Diseño transversal:</b> asociaciones, no causas.<br><b>Población de EE. UU.:</b> usar en Colombia exige "
             "validación externa.<br><b>Sin calibración ni auditoría de equidad</b> por subgrupos (trabajo futuro).", "warn")
    with b:
        seccion("Reproducir los resultados")
        st.code("pip install -r requirements.txt      # o: conda env create -f environment.yml\npython -m src.entrenar_modelos      # reentrena y verifica métricas\n"
                "streamlit run dashboard/app.py", language="bash")
        v = meta["versiones"]
        st.caption(f"Modelos generados con scikit-learn {v['scikit-learn']}, XGBoost {v['xgboost']}, pandas {v['pandas']} y NumPy {v['numpy']}.")
