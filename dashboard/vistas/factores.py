"""Explorador: factores con y sin diabetes, rutas de categorías paralelas, explorador por variable y comparación por sexo."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (AMBAR, BINARIAS, CIAN, EDAD, EDUC, INDIGO_APAGADO, INGRESO, LIMA, MAGENTA, SALUD, SEXO, TEXTO, TEXTO_2, VAR_ES,
                   VIOLETA, chip_filtros, degradado_barras, es, estilo, fuente, grafico, hero, nota, panel, seccion, titulo_grafico)

ETQ = {"Age": EDAD, "GenHlth": SALUD, "Education": EDUC, "Income": INGRESO, "Sex": SEXO}
EXPLORABLES = ["BMI", "Age", "GenHlth", "Income", "Education", "MentHlth", "PhysHlth", "CardioRisk", "Sex"] + BINARIAS
C_SEXO = {0: VIOLETA, 1: CIAN}


def vista():
    df, dff = st.session_state["_df"], st.session_state["_dff"]
    hero("Explorador de factores", "Lo que acompaña a la diabetes",
         "Compare la diabetes entre quienes tienen y no tienen cada factor, siga las rutas que combinan varios factores y explore cualquier "
         "variable. Son asociaciones de una encuesta transversal, no relaciones de causa y efecto.")
    if len(dff) < 200:
        nota("El filtro deja menos de 200 personas; amplíe los criterios del panel lateral.", "warn")
        return
    chip_filtros(len(dff), len(df))

    filas = []
    for v in BINARIAS:
        g = dff.groupby(v).Diabetes_binary.agg(["mean", "size"])
        if set(g.index) >= {0, 1} and g["size"].min() >= 30:
            filas.append({"var": VAR_ES[v], "cod": v, "sin": 100 * g.loc[0, "mean"], "con": 100 * g.loc[1, "mean"],
                          "n_con": g.loc[1, "size"], "rr": g.loc[1, "mean"] / g.loc[0, "mean"]})
    t = pd.DataFrame(filas).sort_values("rr")
    top = t[t.cod != "CholCheck"].iloc[-1]
    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        with panel("dumb"):
            titulo_grafico(f"Con {top['var'].lower()}, la diabetes es {es(top.rr, 1)} veces más frecuente",
                           "Prevalencia sin el factor (círculo) y con el factor (punto brillante)")
            fig = go.Figure()
            for _, r in t.iterrows():
                c_ = MAGENTA if r.rr >= 1 else LIMA
                fig.add_trace(go.Scatter(x=[r.sin, r.con], y=[r["var"]] * 2, mode="lines", line=dict(color=c_, width=3), opacity=.45, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=t.sin, y=t["var"], mode="markers", name="sin el factor", marker=dict(size=11, color="#0a0f26", line=dict(color=TEXTO_2, width=2)),
                                     hovertemplate="%{y}<br>sin el factor: %{x:.1f} %<extra></extra>"))
            fig.add_trace(go.Scatter(x=t.con, y=t["var"], mode="markers+text", name="con el factor",
                                     marker=dict(size=14, color=[MAGENTA if r >= 1 else LIMA for r in t.rr], line=dict(color="#fff", width=1)),
                                     text=[f"  ×{es(v, 1)}" if v >= 1 else f"×{es(v, 2)}  " for v in t.rr],
                                     textposition=["middle right" if v >= 1 else "middle left" for v in t.rr], textfont=dict(size=12, color=TEXTO),
                                     customdata=t.n_con, hovertemplate="%{y}<br>con el factor: %{x:.1f} %<br>%{customdata:,.0f} personas<extra></extra>"))
            fig.update_xaxes(range=[0, t.con.max() * 1.3], ticksuffix=" %"); fig.update_yaxes(showgrid=False, tickfont=dict(color=TEXTO))
            grafico(estilo(fig, 430), key="dumb")
    with c2:
        with panel("parcats"):
            titulo_grafico("Rutas hacia la diabetes", "Cada franja es un grupo de personas que comparte las mismas respuestas. Pase el cursor o arrastre las categorías.")
            m = dff.sample(min(len(dff), 25000), random_state=1)
            dims = [dict(label="Presión alta", values=np.where(m.HighBP == 1, "sí", "no")),
                    dict(label="Colesterol", values=np.where(m.HighChol == 1, "sí", "no")),
                    dict(label="Salud", values=np.where(m.GenHlth >= 4, "regular o mala", "buena o mejor")),
                    dict(label="Ejercicio", values=np.where(m.PhysActivity == 1, "sí", "no")),
                    dict(label="Diabetes", values=np.where(m.Diabetes_binary == 1, "sí", "no"))]
            fig = go.Figure(go.Parcats(dimensions=dims, line=dict(color=m.Diabetes_binary.values, colorscale=[[0, "rgba(70,85,138,.55)"], [1, MAGENTA]], shape="hspline"),
                                       hoveron="color", hoverinfo="count+probability", arrangement="freeform",
                                       labelfont=dict(color="#fff", size=13), tickfont=dict(color=TEXTO_2, size=11.5)))
            fig.update_layout(margin=dict(l=40, r=40, t=34, b=14))
            grafico(estilo(fig, 400), key="parcats")
    nota(f"El control de colesterol aparece arriba por un <b>sesgo de detección</b>: quien va más al médico recibe más diagnósticos. "
         "En las rutas, las franjas magenta se concentran en quienes combinan hipertensión, colesterol alto y salud regular o mala.")

    seccion("Explore una variable", "Elija una variable: a la izquierda, cómo se reparte la población; a la derecha, cuánta diabetes hay en cada nivel.")
    c0, _ = st.columns([1, 2])
    var = c0.selectbox("Variable", EXPLORABLES, format_func=lambda v: VAR_ES[v], index=0, label_visibility="collapsed")
    d = dff[[var, "Diabetes_binary"]].copy()
    if var == "BMI":
        d[var] = pd.cut(d[var], [11, 18.5, 22, 25, 27.5, 30, 32.5, 35, 40, 45, 101], right=False)
        orden = list(d[var].cat.categories)
        nombre = lambda c: f"{c.left:g}–{c.right:g}" if c.right < 101 else f"≥ {c.left:g}"
    elif var in ("MentHlth", "PhysHlth"):
        d[var] = pd.cut(d[var], [-1, 0, 2, 5, 13, 29, 30], labels=["0", "1–2", "3–5", "6–13", "14–29", "30"])
        orden = list(d[var].cat.categories); nombre = str
    else:
        orden = sorted(d[var].unique()); mapa = ETQ.get(var, {0: "No", 1: "Sí"} if var in BINARIAS else {})
        nombre = lambda c, mapa=mapa: mapa.get(c, str(c))
    g = d.groupby(var, observed=True).Diabetes_binary.agg(["mean", "size"]).reindex(orden).dropna()
    x = [nombre(c).replace(" ", "<br>", 1) if len(nombre(c)) > 12 else nombre(c) for c in g.index]
    a, b = st.columns(2, gap="medium")
    with a:
        with panel("dist"):
            titulo_grafico("Cómo se reparte la población", "Porcentaje de cada grupo en cada nivel")
            comp = d.groupby(["Diabetes_binary", var], observed=True).size().unstack(0).reindex(g.index).fillna(0)
            comp = 100 * comp / comp.sum()
            fig = go.Figure()
            if 0 in comp:
                fig.add_trace(go.Bar(x=x, y=comp[0], name="sin diabetes", marker=dict(color=INDIGO_APAGADO, cornerradius=4)))
            if 1 in comp:
                fig.add_trace(go.Bar(x=x, y=comp[1], name="con diabetes", marker=dict(color=MAGENTA, cornerradius=4)))
            fig.update_layout(barmode="group", bargap=.25, bargroupgap=.06); fig.update_yaxes(ticksuffix=" %")
            grafico(estilo(fig, 340, leyenda=True), key="dist")
    with b:
        with panel("prevvar"):
            titulo_grafico("Cuánta diabetes hay en cada nivel", "Porcentaje con diabetes")
            y = 100 * g["mean"].values
            fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers+text", line=dict(color=CIAN, width=3, shape="spline", smoothing=.6),
                                       fill="tozeroy", fillcolor="rgba(34,211,238,.10)", marker=dict(size=10, color=degradado_barras(len(x), CIAN, MAGENTA), line=dict(color="#fff", width=1)),
                                       text=[es(v, 0) + "%" for v in y], textposition="top center", textfont=dict(size=11, color=TEXTO), cliponaxis=False,
                                       customdata=g["size"], hovertemplate="%{x}<br>%{y:.1f} % con diabetes<br>%{customdata:,.0f} personas<extra></extra>"))
            fig.update_yaxes(range=[0, y.max() * 1.28], ticksuffix=" %")
            grafico(estilo(fig, 340), key="prevvar")

    gs = dff.groupby(["Sex", "Age"]).Diabetes_binary.agg(["mean", "size"]).reset_index()
    gs = gs[gs["size"] >= 30]
    piv = gs.pivot(index="Age", columns="Sex", values="mean")
    sub = "Porcentaje con diabetes por edad."
    if {0, 1} <= set(piv.columns) and piv.dropna().shape[0] >= 3:
        dif = 100 * (piv[1] - piv[0]).dropna()
        a_max = dif.idxmax() if abs(dif.max()) >= abs(dif.min()) else dif.idxmin()
        quien = "los hombres" if dif[a_max] > 0 else "las mujeres"
        sub = f"La mayor diferencia aparece entre los {EDAD[a_max]} años: {quien} reportan {es(abs(dif[a_max]), 1)} puntos más de diabetes."
    seccion("Mujeres y hombres", sub)
    with panel("sexo"):
        fig = go.Figure()
        for s_ in (0, 1):
            h = gs[gs.Sex == s_]
            if len(h):
                fig.add_trace(go.Scatter(x=[EDAD[a] for a in h.Age], y=100 * h["mean"], name=("Mujeres", "Hombres")[s_], mode="lines+markers",
                                         line=dict(color=C_SEXO[s_], width=3, shape="spline", smoothing=.5), marker=dict(size=8),
                                         fill="tozeroy", fillcolor=("rgba(167,139,250,.08)", "rgba(34,211,238,.08)")[s_], customdata=h["size"],
                                         hovertemplate=f"{('Mujeres', 'Hombres')[s_]}, %{{x}} años<br>%{{y:.1f}} %<br>%{{customdata:,.0f}} personas<extra></extra>"))
        fig.update_yaxes(rangemode="tozero", ticksuffix=" %"); fig.update_layout(hovermode="x unified")
        grafico(estilo(fig, 360, leyenda=True), key="sexo")
    nota("La salud que cada persona reporta es la señal más fuerte del modelo, pero en parte es <b>consecuencia</b> de la enfermedad: quien ya vive con "
         "diabetes tiende a calificar peor su salud. Se usa como marcador de riesgo, no como causa.", "warn")
    fuente("Fuente: CDC, BRFSS 2023. Cálculos propios sobre la población de la vista.")
