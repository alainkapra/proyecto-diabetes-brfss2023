"""Dashboard interactivo · Predicción de diabetes y segmentación de perfiles de riesgo (BRFSS 2023).

Ejecutar desde la raíz del repositorio:
    streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(page_title="DiabetesIA · BRFSS 2023", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")

from utils import CLUSTER_NOMBRE, CSS, EDAD, EDUC, SALUD, SEXO, cargar_datos, es, filtrar  # noqa: E402
from vistas import factores, metodologia, modelos, panorama, prediccion, segmentacion  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)
st.session_state["_k"] = 0
df = cargar_datos()

CLAVES = ["f_anio", "f_sexo", "f_edad", "f_imc", "f_salud", "f_ingreso", "f_educ", "f_cardio", "f_cluster", "f_diab"]


def restablecer():
    for k in CLAVES:
        st.session_state.pop(k, None)


with st.sidebar:
    st.markdown('<p class="lat-t">Filtros dinámicos</p><p class="lat-s">Recortan la población en todas las páginas. '
                'Los modelos no cambian: son los entrenados en la Fase 3.</p>', unsafe_allow_html=True)
    cont = st.empty()
    anios = sorted(df.Year.unique().tolist())
    if len(anios) > 1:
        st.slider("Año de la encuesta", min(anios), max(anios), (min(anios), max(anios)), key="f_anio")
    else:
        st.slider("Año de la encuesta", 2015, 2023, (anios[0], anios[0]), key="f_anio", disabled=True,
                  help="El conjunto analizado corresponde al ciclo 2023. Si se agregan a data/raw archivos de otros años "
                       "(cdc_brfss_diabetes_AAAA.csv) el filtro se activa automáticamente.")
    st.select_slider("Edad", options=list(EDAD.values()), value=("18–24", "80+"), key="f_edad")
    st.slider("Índice de masa corporal (kg/m²)", 12, 100, (12, 100), key="f_imc")
    st.multiselect("Sexo", list(SEXO.values()), key="f_sexo", placeholder="Mujeres y hombres")
    st.multiselect("Salud general que reporta", list(SALUD.values()), key="f_salud", placeholder="Cualquiera")
    with st.expander("Más criterios", expanded=False):
        st.slider("Nivel de ingreso (1 = más bajo)", 1, 7, (1, 7), key="f_ingreso")
        st.multiselect("Nivel educativo", list(EDUC.values()), key="f_educ", placeholder="Cualquiera")
        st.slider("Condiciones cardiometabólicas", 0, 4, (0, 4), key="f_cardio",
                  help="Suma de hipertensión, colesterol alto, enfermedad coronaria y accidente cerebrovascular.")
        st.multiselect("Perfil de riesgo", list(CLUSTER_NOMBRE.values()), key="f_cluster", placeholder="Los tres")
        st.radio("Diagnóstico", ["Todas", "Con diabetes", "Sin diabetes"], key="f_diab", horizontal=True)
    st.button("Quitar filtros", on_click=restablecer, icon=":material/restart_alt:")
    st.markdown('<p class="lat-s" style="margin-top:1.4rem">Datos: CDC, Behavioral Risk Factor Surveillance System 2023 '
                '(versión de Kaggle de Van Thieu). Proyecto Integrador de Ciencia de Datos · Alain Alexander Camacho Prado.</p>',
                unsafe_allow_html=True)

dff = filtrar(df)
st.session_state["_df"] = df
st.session_state["_dff"] = dff
pct = 100 * len(dff) / len(df)
cont.markdown(f'<div class="lat-n">{es(len(dff), 0)}<small>personas en la vista · {es(pct, 1)} % de la encuesta</small></div>'
              f'<div class="lat-b"><i style="width:{pct:.1f}%"></i></div>', unsafe_allow_html=True)

paginas = [
    st.Page(panorama.vista, title="Panorama", icon=":material/space_dashboard:", url_path="panorama", default=True),
    st.Page(factores.vista, title="Explorador", icon=":material/query_stats:", url_path="factores"),
    st.Page(prediccion.vista, title="Predicción en vivo", icon=":material/bolt:", url_path="prediccion"),
    st.Page(segmentacion.vista, title="Segmentación", icon=":material/scatter_plot:", url_path="segmentacion"),
    st.Page(modelos.vista, title="Desempeño del modelo", icon=":material/speed:", url_path="modelos"),
    st.Page(metodologia.vista, title="Método", icon=":material/schema:", url_path="metodologia"),
]
st.navigation(paginas, position="top").run()

st.markdown('<p class="pie">Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial · Maestría en Inteligencia Artificial · '
            'Universidad de La Salle, Bogotá · 2026. Herramienta académica de tamizaje: no reemplaza el diagnóstico médico.</p>',
            unsafe_allow_html=True)
