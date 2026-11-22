"""Panorama: KPI ejecutivos animados, exploración con selección cruzada, gráfico animado por edad, sunburst y mapa de calor."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (AMBAR, C_CLUSTER, CAT_IMC, CIAN, CLUSTER_NOMBRE, EDAD, INDIGO_APAGADO, LIMA, MAGENTA, SECUENCIAL, SEXO,
                   TEXTO, TEXTO_2, VIOLETA, cargar_modelos, chip_filtros, degradado_barras, es, estilo, fuente, grafico, hero, kpis,
                   nota, panel, seccion, titulo_grafico)


def tabla(d, col):
    return d.groupby(col, observed=True).Diabetes_binary.agg(prev="mean", n="size", casos="sum")


def barras(t, x, colores, alto=300, sel=None):
    y = 100 * t["prev"].values
    if sel:
        colores = [c if xi in sel else "#2a3160" for c, xi in zip(colores, x)]
    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=colores, line=dict(width=0), cornerradius=6),
                           selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
                           text=[es(v, 0) + "%" for v in y], textposition="outside", textfont=dict(size=11, color=TEXTO), cliponaxis=False,
                           customdata=np.c_[t["n"], t["casos"]],
                           hovertemplate="<b>%{x}</b><br>%{y:.1f} % con diabetes<br>%{customdata[0]:,.0f} personas<extra></extra>"))
    fig.update_yaxes(visible=False, range=[0, y.max() * 1.22])
    fig.update_layout(clickmode="event+select", dragmode="select")
    return estilo(fig, alto)


def vista():
    df, dff = st.session_state["_df"], st.session_state["_dff"]
    _, _, _, meta = cargar_modelos()
    hero("Panorama · KPI ejecutivos", "Diabetes y perfiles de riesgo en 272.769 adultos",
         "Indicadores de la población y del modelo de aprendizaje automático. Todo responde a los filtros del panel lateral, "
         "y los gráficos de esta página se filtran entre sí: seleccione un rango de edades para enfocar el análisis.")
    if len(dff) < 30:
        nota("El filtro deja menos de 30 personas. Amplíe los criterios del panel lateral.", "warn")
        return
    chip_filtros(len(dff), len(df))
    prev = dff.Diabetes_binary.mean()
    kpis([
        dict(etiqueta="Personas en la vista", valor=len(dff), dec=0, icono="👥", color=CIAN, barra=100 * len(dff) / len(df),
             nota=f"{es(dff.Diabetes_binary.sum(), 0)} reportan diabetes"),
        dict(etiqueta="Prevalencia de diabetes", valor=100 * prev, dec=1, sufijo="%", icono="🩸", color=MAGENTA, barra=100 * prev / .45,
             nota=f"total de la encuesta: {es(100 * df.Diabetes_binary.mean(), 1)} %"),
        dict(etiqueta="IMC promedio", valor=dff.BMI.mean(), dec=1, icono="⚖️", color=AMBAR, barra=(dff.BMI.mean() - 18) / 22 * 100,
             nota=f"{es(100 * (dff.BMI >= 30).mean(), 0)} % con obesidad"),
        dict(etiqueta="Hipertensión", valor=100 * dff.HighBP.mean(), dec=0, sufijo="%", icono="❤️", color=VIOLETA, barra=100 * dff.HighBP.mean(),
             nota=f"colesterol alto: {es(100 * dff.HighChol.mean(), 0)} %"),
        dict(etiqueta="Dos o más condiciones", valor=100 * (dff.CardioRisk >= 2).mean(), dec=0, sufijo="%", icono="🫀", color=LIMA,
             barra=100 * (dff.CardioRisk >= 2).mean(), nota="cardiometabólicas a la vez"),
    ], alto=168)
    m = meta["metricas_prueba_clasificacion"]
    pf = df[df.Cluster > 0].groupby("Cluster").Diabetes_binary.mean()
    kpis([
        dict(etiqueta="AUC-ROC del modelo", valor=m["auc_roc"], dec=3, icono="🎯", color=CIAN, barra=100 * m["auc_roc"], nota="XGBoost · 40.916 personas de prueba"),
        dict(etiqueta="AUC-PR", valor=m["auc_pr"], dec=3, icono="📈", color=VIOLETA, barra=100 * m["auc_pr"], nota="línea base = 0,146 (prevalencia)"),
        dict(etiqueta="Sensibilidad de tamizaje", valor=100 * m["recall_05"], dec=1, sufijo="%", icono="🔎", color=LIMA, barra=100 * m["recall_05"],
             nota="diabéticos detectados (umbral 0,5)"),
        dict(etiqueta="Valor predictivo positivo", valor=100 * m["precision_05"], dec=1, sufijo="%", icono="🧪", color=AMBAR, barra=100 * m["precision_05"],
             nota="de los señalados tiene diabetes"),
        dict(etiqueta="Perfiles de riesgo", valor=3, dec=0, icono="🧩", color=MAGENTA, barra=100 * pf.max(),
             nota=f"prevalencias de {es(100 * pf.min(), 0)} % a {es(100 * pf.max(), 0)} %"),
    ], alto=168)

    seccion("Explore por edad", "Arrastre el cursor sobre las barras de edad para seleccionar un rango: los gráficos de la derecha se recalculan al instante para esas edades. Doble clic para volver a todas.")
    t_age = tabla(dff, "Age")
    x_age = [EDAD[i] for i in t_age.index]
    c1, c2 = st.columns([1.25, 1], gap="medium")
    seleccion = (st.session_state.get("sel_edad") or {}).get("selection")
    sel = set()
    if seleccion:
        sel |= {p_["x"] for p_ in seleccion.get("points", []) if p_.get("x") in x_age}
        for caja in seleccion.get("box", []):                       # arrastre: rango de categorías en el eje x
            x0, x1 = sorted(caja["x"][:2])
            sel |= {x_age[i] for i in range(len(x_age)) if x0 - .5 <= i <= x1 + .5}
    sel = sorted(sel, key=x_age.index)
    with c1:
        with panel("edad"):
            titulo_grafico("Prevalencia de diabetes por grupo de edad", "Arrastre para seleccionar un rango de edades")
            grafico(barras(t_age, x_age, degradado_barras(len(x_age), CIAN, MAGENTA), 360, sel),
                    on_select="rerun", selection_mode=("points", "box"), key="sel_edad")
    inv = {v: k for k, v in EDAD.items()}
    dsel = dff[dff.Age.isin([inv[s] for s in sel])] if sel else dff
    with c2:
        with panel("detalle"):
            etiqueta = ("edades " + ", ".join(sel)) if sel else "todas las edades"
            titulo_grafico(f"Detalle: {etiqueta}", f"{es(len(dsel), 0)} personas · {es(100 * dsel.Diabetes_binary.mean(), 1)} % con diabetes")
            t = tabla(dsel, "BMI_cat")
            fig = barras(t, [c.replace(" ", "<br>") for c in t.index], degradado_barras(len(t), AMBAR, MAGENTA), 175)
            grafico(fig, key="det_imc")
            t = tabla(dsel, "CardioRisk")
            fig = barras(t, [f"{i} cond." for i in t.index], degradado_barras(len(t), VIOLETA, MAGENTA), 175)
            grafico(fig, key="det_cr")
    if sel:
        fuente("Doble clic sobre el gráfico de edades para quitar la selección.")

    seccion("El riesgo en movimiento", "Pulse ▶ para ver cómo cambia la prevalencia en cada categoría de peso a medida que avanza la edad.")
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        with panel("anim"):
            p = dff.groupby(["Age", "BMI_cat"], observed=True).Diabetes_binary.agg(["mean", "size"]).reset_index()
            p = p[p["size"] >= 40]
            edades = sorted(p.Age.unique())
            col = degradado_barras(len(CAT_IMC), CIAN, MAGENTA)
            def barra_edad(a):
                q = p[p.Age == a].set_index("BMI_cat").reindex(CAT_IMC)
                return go.Bar(x=CAT_IMC, y=100 * q["mean"], marker=dict(color=col, cornerradius=6),
                              text=[("" if np.isnan(v) else es(v, 0) + "%") for v in 100 * q["mean"]], textposition="outside",
                              textfont=dict(color=TEXTO, size=12), hovertemplate="%{x}: %{y:.1f} %<extra></extra>")
            if edades:
                fig = go.Figure(data=[barra_edad(edades[0])], frames=[go.Frame(data=[barra_edad(a)], name=EDAD[a]) for a in edades])
                fig.update_layout(
                    updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=1.16, xanchor="left", yanchor="bottom", pad=dict(r=8, t=0), bgcolor="rgba(34,211,238,.15)",
                                      bordercolor=CIAN, font=dict(color=TEXTO),
                                      buttons=[dict(label="▶  Reproducir", method="animate",
                                                    args=[None, dict(frame=dict(duration=650, redraw=True), transition=dict(duration=450, easing="cubic-in-out"), fromcurrent=True)]),
                                               dict(label="❚❚", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
                    sliders=[dict(active=0, x=0, len=1, y=0, yanchor="top", pad=dict(t=46, b=0), currentvalue=dict(prefix="Edad: ", font=dict(color=CIAN, size=13)),
                                  font=dict(color=TEXTO_2, size=10), bgcolor="rgba(255,255,255,.08)", activebgcolor=CIAN, bordercolor="rgba(0,0,0,0)",
                                  steps=[dict(label=EDAD[a], method="animate", args=[[EDAD[a]], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=300))]) for a in edades])])
                fig.update_yaxes(range=[0, 100 * p["mean"].max() * 1.18], ticksuffix=" %")
                estilo(fig, 430); fig.update_layout(margin=dict(t=56, b=110))
                grafico(fig, key="anim")
    with c2:
        with panel("sun"):
            titulo_grafico("Población por perfil, sexo y diagnóstico", "Haga clic en un sector para entrar; clic en el centro para volver")
            s = dff[dff.Cluster > 0].groupby(["Cluster", "Sex", "Diabetes_binary"]).size().reset_index(name="n")
            ids, labels, parents, values, colors = [], [], [], [], []
            for k in (1, 2, 3):
                sk = s[s.Cluster == k]
                if not len(sk): continue
                ids.append(f"c{k}"); labels.append(CLUSTER_NOMBRE[k]); parents.append(""); values.append(int(sk.n.sum())); colors.append(C_CLUSTER[k])
                for sx in (0, 1):
                    ss = sk[sk.Sex == sx]
                    if not len(ss): continue
                    ids.append(f"c{k}s{sx}"); labels.append(SEXO[sx] + ("es" if sx == 0 else "s")); parents.append(f"c{k}"); values.append(int(ss.n.sum()))
                    colors.append(C_CLUSTER[k] + "99")
                    for dg in (0, 1):
                        nn = int(ss[ss.Diabetes_binary == dg].n.sum())
                        if nn:
                            ids.append(f"c{k}s{sx}d{dg}"); labels.append("con diabetes" if dg else "sin diabetes"); parents.append(f"c{k}s{sx}")
                            values.append(nn); colors.append(MAGENTA if dg else INDIGO_APAGADO)
            fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                                        marker=dict(colors=colors, line=dict(color="#0a0f26", width=2)), insidetextorientation="horizontal", maxdepth=2,
                                        textfont=dict(color="#fff", size=12), hovertemplate="<b>%{label}</b><br>%{value:,.0f} personas<br>%{percentParent:.1%} del nivel superior<extra></extra>"))
            fig.update_layout(margin=dict(t=6, l=6, r=6, b=6))
            grafico(estilo(fig, 400), key="sun")

    seccion("Donde se cruzan la edad y el peso", "Porcentaje con diabetes en cada combinación. Se ocultan las celdas con menos de 50 personas.")
    with panel("heat"):
        p = dff.pivot_table(index="BMI_cat", columns="Age", values="Diabetes_binary", aggfunc=["mean", "size"], observed=False)
        prev_m, n_m = p["mean"].reindex(index=CAT_IMC, columns=range(1, 14)), p["size"].reindex(index=CAT_IMC, columns=range(1, 14))
        z = (100 * prev_m).where(n_m >= 50)
        fig = go.Figure(go.Heatmap(z=z.values, x=[EDAD[c] for c in z.columns], y=list(z.index), colorscale=[[i / 6, c] for i, c in enumerate(SECUENCIAL)],
                                   zmin=0, zmax=45, xgap=3, ygap=3, text=[[("" if np.isnan(v) else es(v, 0)) for v in fila] for fila in z.values],
                                   texttemplate="%{text}", textfont=dict(size=12, color="#fff"), customdata=n_m.fillna(0).values,
                                   colorbar=dict(title=dict(text="% con<br>diabetes", font=dict(color=TEXTO_2, size=11)), tickfont=dict(color=TEXTO_2), thickness=10, outlinewidth=0),
                                   hovertemplate="%{y} · %{x} años<br>%{z:.1f} % con diabetes<br>%{customdata:,.0f} personas<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        grafico(estilo(fig, 340), key="heat")
    t_hi = dff[(dff.Age >= 9) & (dff.BMI >= 35)].Diabetes_binary
    t_lo = dff[(dff.Age <= 4) & (dff.BMI < 25)].Diabetes_binary
    if len(t_hi) > 50 and len(t_lo) > 50:
        nota(f"Entre los adultos de 60 años o más con obesidad grado II o III, <b>{es(100 * t_hi.mean(), 0)} de cada 100</b> tienen diabetes; "
             f"entre los menores de 40 con peso normal, apenas <b>{es(100 * t_lo.mean(), 1)}</b>. La edad amplifica el efecto del peso: por eso la "
             "interacción edad × IMC es una de las tres variables que más pesan en el modelo.")
    fuente("Fuente: CDC, BRFSS 2023. Cálculos propios sobre la población de la vista.")
