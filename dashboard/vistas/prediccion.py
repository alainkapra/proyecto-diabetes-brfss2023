"""Predicción en vivo: el usuario describe un perfil y obtiene al instante el puntaje del XGBoost (indicador animado),
su lugar en la población, la explicación en cascada SHAP, el perfil asignado y escenarios «¿qué pasaría si…?»."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from utils import (AMBAR, C_CLUSTER, CIAN, CLUSTER_DESC, CLUSTER_NOMBRE, EDAD, EDUC, ENTRADAS, INDIGO_APAGADO, LIMA, MAGENTA, SALUD,
                   TEXTO, TEXTO_2, TEXTO_3, VAR_ES, VIOLETA, asignar_cluster, cargar_modelos, cargar_predicciones, es, estilo, grafico,
                   fuente, hero, nota, panel, seccion, titulo_grafico)
from src.preprocesamiento import CL_VARS, vars_cluster

POR_DEFECTO = dict(p_edad="50–54", p_sexo="Mujer", p_peso=78.0, p_talla=165.0, p_salud="Buena", p_fis=2, p_men=2,
                   p_educ="Secundaria completa", p_ing=4, p_hbp=False, p_chol=False, p_cc=True, p_smk=False, p_stk=False,
                   p_hd=False, p_act=True, p_cost=False, p_walk=False)
EJEMPLOS = {
    "Bajo riesgo": dict(p_edad="30–34", p_sexo="Mujer", p_peso=60.0, p_talla=165.0, p_salud="Muy buena", p_fis=0, p_men=1,
                        p_educ="Universitaria completa", p_ing=7, p_hbp=False, p_chol=False, p_cc=True, p_smk=False, p_stk=False,
                        p_hd=False, p_act=True, p_cost=False, p_walk=False),
    "Riesgo silencioso": dict(p_edad="65–69", p_sexo="Hombre", p_peso=92.0, p_talla=175.0, p_salud="Buena", p_fis=1, p_men=0,
                              p_educ="Secundaria completa", p_ing=5, p_hbp=True, p_chol=True, p_cc=True, p_smk=True, p_stk=False,
                              p_hd=False, p_act=True, p_cost=False, p_walk=False),
    "Alto riesgo": dict(p_edad="60–64", p_sexo="Mujer", p_peso=105.0, p_talla=160.0, p_salud="Regular", p_fis=15, p_men=10,
                        p_educ="Hasta secundaria incompleta", p_ing=2, p_hbp=True, p_chol=True, p_cc=True, p_smk=False, p_stk=False,
                        p_hd=True, p_act=False, p_cost=True, p_walk=True),
}


def cargar_ejemplo(nombre):
    st.session_state.update(EJEMPLOS[nombre])


def perfil_desde_estado(s) -> pd.DataFrame:
    inv_e = {v: k for k, v in EDAD.items()}; inv_s = {v: k for k, v in SALUD.items()}; inv_ed = {v: k for k, v in EDUC.items()}
    imc = s.p_peso / (s.p_talla / 100) ** 2
    fila = {"HighBP": int(s.p_hbp), "HighChol": int(s.p_chol), "CholCheck": int(s.p_cc), "BMI": int(round(imc)),
            "Smoker": int(s.p_smk), "Stroke": int(s.p_stk), "HeartDiseaseorAttack": int(s.p_hd), "PhysActivity": int(s.p_act),
            "NoDocbcCost": int(s.p_cost), "GenHlth": inv_s[s.p_salud], "MentHlth": int(s.p_men), "PhysHlth": int(s.p_fis),
            "DiffWalk": int(s.p_walk), "Sex": 1 if s.p_sexo == "Hombre" else 0, "Age": inv_e[s.p_edad],
            "Education": inv_ed[s.p_educ], "Income": int(s.p_ing)}
    return pd.DataFrame([fila])[ENTRADAS]


def contribuciones(clf, X: pd.DataFrame):
    """Contribuciones SHAP exactas (TreeSHAP nativo de XGBoost) en escala log-odds."""
    Xt = clf[:-1].transform(X)
    c = clf[-1].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0]
    return pd.Series(c[:-1], index=Xt.columns), Xt.iloc[0], float(c[-1])


def valor_legible(var, v):
    if var == "Age": return f"{EDAD[int(v)]} años"
    if var == "GenHlth": return SALUD[int(v)].lower()
    if var == "Sex": return "hombre" if v else "mujer"
    if var == "Education": return EDUC.get(int(v), str(int(v))).lower()
    if var == "Income": return f"nivel {int(v)}"
    if var in ("MentHlth", "PhysHlth", "UnhealthyDays"): return f"{int(v)} días"
    if var in ("BMI", "Age_BMI"): return es(v, 0)
    if var == "CardioRisk": return f"{int(v)}"
    return "sí" if v else "no"


def vista():
    clf, _, clu, meta = cargar_modelos()
    umbral = meta["umbral_f1"]
    for k, v in POR_DEFECTO.items():
        st.session_state.setdefault(k, v)
    hero("Módulo de predicción en vivo", "¿Cuál sería el riesgo de esta persona?",
         "Describa a un adulto: el modelo XGBoost de la Fase 3 calcula su puntaje al instante, lo ubica en la población, explica qué lo sube o lo baja "
         "y simula qué pasaría si cambiaran factores modificables. Cada cambio en el formulario actualiza el resultado.")
    c = st.columns([1, 1.05, 1.3, 1.05, 1.8])
    c[0].markdown(f'<p style="color:{TEXTO_2};font-size:.9rem;padding-top:.5rem">Cargar un ejemplo:</p>', unsafe_allow_html=True)
    c[1].button("Bajo riesgo", on_click=cargar_ejemplo, args=("Bajo riesgo",), width="stretch", icon=":material/eco:")
    c[2].button("Riesgo silencioso", on_click=cargar_ejemplo, args=("Riesgo silencioso",), width="stretch", icon=":material/visibility_off:")
    c[3].button("Alto riesgo", on_click=cargar_ejemplo, args=("Alto riesgo",), width="stretch", icon=":material/local_fire_department:")

    izq, der = st.columns([1.02, 1.38], gap="large")
    with izq:
        with st.container(key="ficha"):
            st.markdown('<p class="ficha-t">👤 Datos personales</p>', unsafe_allow_html=True)
            a, b = st.columns(2)
            a.select_slider("Edad", options=list(EDAD.values()), key="p_edad")
            b.radio("Sexo", ["Mujer", "Hombre"], key="p_sexo", horizontal=True)
            a, b = st.columns(2)
            a.number_input("Peso (kg)", 30.0, 250.0, step=1.0, format="%.0f", key="p_peso")
            b.number_input("Estatura (cm)", 120.0, 220.0, step=1.0, format="%.0f", key="p_talla")
            imc = st.session_state.p_peso / (st.session_state.p_talla / 100) ** 2
            cat = ("bajo peso" if imc < 18.5 else "peso normal" if imc < 25 else "sobrepeso" if imc < 30 else
                   "obesidad grado I" if imc < 35 else "obesidad grado II" if imc < 40 else "obesidad grado III")
            st.markdown(f'<p style="font-size:.86rem;color:{CIAN};margin:-.3rem 0 .6rem">IMC {es(imc, 1)} kg/m² · {cat}</p>', unsafe_allow_html=True)
            st.selectbox("Nivel educativo", list(EDUC.values()), key="p_educ")
            st.slider("Ingreso del hogar (1 = más bajo, 7 = más alto)", 1, 7, key="p_ing")
            st.markdown('<p class="ficha-t">🩺 Estado de salud</p>', unsafe_allow_html=True)
            st.select_slider("Cómo califica su salud", options=list(SALUD.values()), key="p_salud")
            a, b = st.columns(2)
            a.slider("Días con malestar físico (mes)", 0, 30, key="p_fis")
            b.slider("Días con malestar emocional (mes)", 0, 30, key="p_men")
            a, b = st.columns(2)
            a.toggle("Hipertensión", key="p_hbp"); b.toggle("Colesterol alto", key="p_chol")
            a.toggle("Infarto", key="p_hd", help="Infarto o enfermedad coronaria."); b.toggle("Derrame", key="p_stk", help="Accidente cerebrovascular (derrame cerebral).")
            a.toggle("Le cuesta caminar", key="p_walk"); b.toggle("Mide colesterol", key="p_cc", help="Se midió el colesterol en los últimos cinco años.")
            st.markdown('<p class="ficha-t">🏃 Hábitos y acceso</p>', unsafe_allow_html=True)
            a, b = st.columns(2)
            a.toggle("Hace ejercicio", key="p_act", help="Actividad física fuera del trabajo en los últimos 30 días.")
            b.toggle("Fumador", key="p_smk", help="Al menos 100 cigarrillos en la vida.")
            a.toggle("Evitó el médico", key="p_cost", help="En el último año no fue al médico por el costo.")

    X = perfil_desde_estado(st.session_state)
    p = float(clf.predict_proba(X)[0, 1])
    pred = cargar_predicciones()
    pct = 100 * (pred.proba < p).mean()
    cerca = pred[(pred.proba - p).abs() <= 0.03]
    obs = cerca.y.mean() if len(cerca) >= 40 else np.nan
    if p >= umbral:
        tono, banda, texto = MAGENTA, "RIESGO ALTO", "Supera el umbral óptimo del modelo: en un programa de tamizaje sería prioridad para una glucosa en ayunas o una HbA1c."
    elif p >= 0.5:
        tono, banda, texto = AMBAR, "RIESGO INTERMEDIO", "Supera el umbral de tamizaje: una prueba de laboratorio confirmaría o descartaría la diabetes."
    else:
        tono, banda, texto = LIMA, "RIESGO BAJO", "Por debajo del umbral de tamizaje: mantener hábitos y controles según la edad."

    with der:
        a, b = st.columns([1, 1.15], gap="small")
        with a:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=100 * p, number=dict(font=dict(size=54, color="#fff", family="Sora"), valueformat=".0f"),
                gauge=dict(shape="angular", axis=dict(range=[0, 100], tickwidth=1, tickcolor=TEXTO_3, tickfont=dict(color=TEXTO_3, size=10)),
                           bar=dict(color=tono, thickness=.32), bgcolor="rgba(255,255,255,.04)", borderwidth=0,
                           steps=[dict(range=[0, 50], color="rgba(163,230,53,.12)"), dict(range=[50, 100 * umbral], color="rgba(251,191,36,.14)"),
                                  dict(range=[100 * umbral, 100], color="rgba(244,114,182,.16)")],
                           threshold=dict(line=dict(color="#fff", width=3), thickness=.9, value=100 * p))))
            estilo(fig, 240); fig.update_layout(margin=dict(l=34, r=34, t=24, b=4))
            grafico(fig, key="gauge")
        with b:
            st.markdown(f'<div class="score" style="--c:{tono}"><div class="lbl">{banda}</div>'
                        f'<p>Puntaje de <b>{es(100 * p, 1)} sobre 100</b>, más alto que el de <b>{es(min(pct, 99.9), 0 if pct < 99 else 1)} de cada 100</b> adultos evaluados. {texto}'
                        + (f' Entre quienes tienen un puntaje parecido, <b>{es(100 * obs, 0)} de cada 100</b> tenían diabetes.' if not np.isnan(obs) else "")
                        + '</p></div>', unsafe_allow_html=True)
        with panel("dist"):
            h, bordes = np.histogram(100 * pred.proba, bins=50, range=(0, 100))
            centros = (bordes[:-1] + bordes[1:]) / 2
            col = [LIMA if c_ < 50 else AMBAR if c_ < 100 * umbral else MAGENTA for c_ in centros]
            fig = go.Figure(go.Bar(x=centros, y=h, marker=dict(color=col, opacity=.55, line=dict(width=0)), width=1.7,
                                   hovertemplate="puntaje %{x:.0f}: %{y:,} personas<extra></extra>"))
            fig.add_vline(x=100 * p, line=dict(color="#fff", width=2.5))
            fig.add_annotation(x=100 * p, y=h.max() * .95, text="<b>este perfil</b>", showarrow=False, font=dict(size=12.5, color="#fff"),
                               xanchor="right" if p > .8 else "left", xshift=-6 if p > .8 else 6, bgcolor="rgba(10,15,38,.7)")
            fig.update_yaxes(visible=False); fig.update_xaxes(range=[0, 100], title="puntaje de las 40.916 personas de prueba")
            grafico(estilo(fig, 165), key="hist")

        contrib, xt, base = contribuciones(clf, X)
        orden = contrib.abs().sort_values(ascending=False).index
        top = contrib.reindex(orden[:7]); resto = contrib.reindex(orden[7:]).sum()
        with panel("cascada"):
            titulo_grafico("Cascada de la predicción (SHAP)", "Desde el punto de partida del modelo, cada factor suma (magenta) o resta (verde) riesgo hasta llegar a este perfil")
            etq = ["Punto de partida"] + [f"{VAR_ES.get(v, v)}: {valor_legible(v, xt[v])}" for v in top.index] + ["Otros factores", "Este perfil"]
            vals = [base] + list(top.values) + [resto, 0]
            fig = go.Figure(go.Waterfall(orientation="h", y=etq, x=vals, measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
                                         increasing=dict(marker=dict(color=MAGENTA)), decreasing=dict(marker=dict(color=LIMA)),
                                         totals=dict(marker=dict(color=VIOLETA)), connector=dict(line=dict(color="rgba(163,174,208,.3)", width=1, dash="dot")),
                                         text=[f"{base:.2f}"] + [f"{v:+.2f}" for v in vals[1:-1]] + [f"{base + sum(vals[1:-1]):.2f}"], textposition="outside",
                                         textfont=dict(color=TEXTO, size=11), hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
            acum = np.cumsum([base] + list(top.values) + [resto])
            lo_, hi_ = min(0, acum.min()), max(0, acum.max())
            pad = (hi_ - lo_) * .28 + .1
            fig.update_xaxes(title="escala log-odds del modelo", zeroline=True, zerolinecolor="rgba(163,174,208,.4)", range=[lo_ - pad, hi_ + pad])
            fig.update_yaxes(tickfont=dict(color=TEXTO, size=12), showgrid=False, autorange="reversed")
            fig.update_traces(cliponaxis=False)
            grafico(estilo(fig, 380), key="cascada")

    k = int(asignar_cluster(X, clu)[0])
    df = st.session_state["_df"]
    pc = df[df.Cluster == k].Diabetes_binary.mean()
    seccion("Perfil poblacional y escenarios")
    a, b = st.columns([1, 1.5], gap="medium")
    with a:
        st.markdown(f'<div class="clc" style="--c:{C_CLUSTER[k]}"><div class="nom">Se parece al perfil «{CLUSTER_NOMBRE[k]}»</div>'
                    f'<div class="fila"><div><div class="big">{es(100 * pc, 1)}%</div><div class="lbl">con diabetes en el perfil</div></div></div>'
                    f'<p>{CLUSTER_DESC[k]}</p></div>', unsafe_allow_html=True)
        base_c = df[df.Cluster > 0]
        z_ = lambda D: (D - base_c[CL_VARS].mean()) / base_c[CL_VARS].std()
        zp = z_(vars_cluster(X, clu["limites_imc"]).iloc[0])
        zc = z_(base_c[base_c.Cluster == k][CL_VARS].mean())
        nombres = ["IMC", "Edad", "Mala salud", "Malestar emocional", "Malestar físico", "Educación", "Ingreso", "Condiciones"]
        with panel("radar"):
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=list(zc.clip(-2, 3) + 2) + [zc.clip(-2, 3).iloc[0] + 2], theta=nombres + [nombres[0]], name="promedio del perfil",
                                          fill="toself", line=dict(color=C_CLUSTER[k], width=2), opacity=.45))
            fig.add_trace(go.Scatterpolar(r=list(zp.clip(-2, 3) + 2) + [zp.clip(-2, 3).iloc[0] + 2], theta=nombres + [nombres[0]], name="esta persona",
                                          line=dict(color="#fff", width=2.5), marker=dict(size=6)))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 5]),
                                         angularaxis=dict(gridcolor="rgba(163,174,208,.18)", tickfont=dict(color=TEXTO_2, size=11))))
            estilo(fig, 330, leyenda=True)
            fig.update_layout(legend=dict(orientation="h", y=-.08, yanchor="top", x=.5, xanchor="center"), margin=dict(l=95, r=95, t=24, b=40))
            grafico(fig, key="radar_pred")
    with b:
        esc = {}
        if X.BMI.iloc[0] >= 25:
            e = X.copy(); e["BMI"] = int(round(X.BMI.iloc[0] * 0.9)); esc["Bajar 10 % del peso"] = e
        if X.PhysActivity.iloc[0] == 0:
            e = X.copy(); e["PhysActivity"] = 1; esc["Hacer ejercicio"] = e
        if X.HighBP.iloc[0] == 1:
            e = X.copy(); e["HighBP"] = 0; esc["Sin hipertensión"] = e
        if X.HighChol.iloc[0] == 1:
            e = X.copy(); e["HighChol"] = 0; esc["Sin colesterol alto"] = e
        if X.Smoker.iloc[0] == 1:
            e = X.copy(); e["Smoker"] = 0; esc["No haber fumado"] = e
        if len(esc) > 1:
            comb = X.copy()
            for e in esc.values():
                for col_ in ENTRADAS:
                    if e[col_].iloc[0] != X[col_].iloc[0]:
                        comb[col_] = e[col_].iloc[0]
            esc["Todos los cambios"] = comb
        with panel("escen"):
            titulo_grafico("¿Qué pasaría si…?", "Del puntaje actual (círculo) al puntaje con cada cambio (flecha)")
            if not esc:
                nota("Este perfil no tiene factores modificables marcados: exceso de peso, inactividad, hipertensión, colesterol alto o tabaquismo.", "ok")
            else:
                P = {n_: 100 * float(clf.predict_proba(e)[0, 1]) for n_, e in esc.items()}
                nombres_e = list(P)[::-1]
                fig = go.Figure()
                for n_ in nombres_e:
                    d = P[n_] - 100 * p
                    col_ = LIMA if d <= -1 else MAGENTA if d >= 1 else TEXTO_3
                    fig.add_annotation(x=P[n_], y=n_, ax=100 * p, ay=n_, xref="x", yref="y", axref="x", ayref="y", showarrow=abs(d) >= 1,
                                       arrowhead=2, arrowsize=1.1, arrowwidth=2.2, arrowcolor=col_, standoff=7, text="")
                    fig.add_annotation(x=max(P[n_], 100 * p), y=n_, xshift=10, text=("igual" if abs(d) < 1 else f"<b>{es(P[n_], 0)}</b> ({'−' if d < 0 else '+'}{es(abs(d), 0)})"),
                                       showarrow=False, xanchor="left", font=dict(size=12, color=col_))
                fig.add_trace(go.Scatter(x=[100 * p] * len(nombres_e), y=nombres_e, mode="markers", marker=dict(size=13, color="#0a0f26", line=dict(color="#fff", width=2)),
                                         hovertemplate="actual: %{x:.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[P[n_] for n_ in nombres_e], y=nombres_e, mode="markers",
                                         marker=dict(size=13, color=[LIMA if P[n_] - 100 * p <= -1 else MAGENTA if P[n_] - 100 * p >= 1 else TEXTO_3 for n_ in nombres_e]),
                                         hovertemplate="%{y}: %{x:.0f}<extra></extra>"))
                fig.add_vline(x=50, line=dict(color=AMBAR, width=1, dash="dot"))
                fig.update_xaxes(range=[0, 112], title="puntaje de riesgo"); fig.update_yaxes(showgrid=False, tickfont=dict(color=TEXTO, size=12.5))
                grafico(estilo(fig, 110 + 46 * len(P)), key="escen")
                if 100 * p > 85:
                    fuente("En perfiles de riesgo muy alto el puntaje se satura y un cambio aislado lo mueve poco.")
    nota("El puntaje ordena el riesgo para decidir a quién hacerle una prueba; no es un diagnóstico. El modelo aprendió de una encuesta autorreportada de "
         "adultos de EE. UU. y detecta cerca de 79 de cada 100 casos. Los escenarios muestran lo que el modelo asocia, no efectos garantizados.", "warn")
