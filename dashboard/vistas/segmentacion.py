"""Segmentación: tres perfiles de K-Means, dispersión 3D/2D en el espacio PCA con giro automático,
perfiles numéricos promedio (radar, mapa y barras) y validación del número de clústeres."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (AMBAR, C_CLUSTER, C_DIAB, CIAN, CLUSTER_DESC, CLUSTER_NOMBRE, DIVERGENTE, EDAD, MAGENTA, TEXTO, TEXTO_2, TEXTO_3,
                   VIOLETA, cargar_modelos, cargar_muestra_pca, cargar_resultados, es, estilo, filtrar, fuente, grafico, hero, nota, panel,
                   retrato, seccion, titulo_grafico)
from src.preprocesamiento import CL_VARS

ET = {"BMI": "IMC", "Age": "Edad", "GenHlth": "Mala salud (1–5)", "MentHlth": "Malestar emocional", "PhysHlth": "Malestar físico",
      "Education": "Educación", "Income": "Ingreso", "CardioRisk": "Condiciones"}


def vista():
    df, dff = st.session_state["_df"], st.session_state["_dff"]
    _, _, clu, meta = cargar_modelos()
    R = cargar_resultados()
    base = df[df.Cluster > 0]
    g = base.groupby("Cluster").agg(n=("BMI", "size"), prev=("Diabetes_binary", "mean"))
    hero("Módulo de segmentación · K-Means", "Tres maneras de estar en riesgo",
         "K-Means agrupó a las personas con ocho variables de salud, edad y nivel socioeconómico, sin conocer quién tenía diabetes. "
         "Después se comprobó que los tres grupos tienen riesgos muy distintos. Gire el mapa 3D, cambie el color y compare los perfiles promedio.")
    cols = st.columns(3, gap="medium")
    for k, c in zip((1, 2, 3), cols):
        n = int(round(100 * g.prev[k]))
        c.markdown(f'<div class="clc" style="--c:{C_CLUSTER[k]}"><div class="nom">C{k} · {CLUSTER_NOMBRE[k]}</div>'
                   f'<div class="fila"><div><div class="big">{es(100 * g.n[k] / g.n.sum(), 0)}%</div><div class="lbl">de la población</div></div>'
                   f'<div><div class="big">{es(100 * g.prev[k], 1)}%</div><div class="lbl">con diabetes</div></div></div>'
                   + retrato([(n, C_CLUSTER[k], C_CLUSTER[k])], total=100, cols=25, r=4.2, paso=11) +
                   f'<p>{CLUSTER_DESC[k]}</p></div>', unsafe_allow_html=True)

    seccion("Mapa interactivo de la población", "12.000 personas proyectadas en las componentes principales; también responde a los filtros del panel lateral. "
            "En 3D, pulse «Girar» para rotar la nube o arrástrela con el ratón.")
    m = cargar_muestra_pca().copy(); m["Year"] = 2023
    mf = filtrar(m)
    a, b, _ = st.columns([1, 1.3, 2])
    dim = a.segmented_control("Vista", ["3D", "2D"], default="3D", key="seg_dim") or "3D"
    color = b.segmented_control("Color", ["perfil", "diagnóstico"], default="perfil", key="seg_col") or "perfil"
    var = meta["pca_varianza"]
    with panel("mapa"):
        if len(mf) < 10:
            nota("No hay personas de la muestra con los filtros actuales.", "warn")
        else:
            fig = go.Figure()
            grupos = ([(mf[mf.Cluster == k], C_CLUSTER[k], f"C{k} · {CLUSTER_NOMBRE[k]}") for k in (1, 2, 3)] if color == "perfil" else
                      [(mf[mf.Diabetes_binary == k], C_DIAB[k], "con diabetes" if k else "sin diabetes") for k in (0, 1)])
            for d, col, nom in grupos:
                cd = np.c_[d.BMI, d.Age.map(EDAD), d.GenHlth, d.CardioRisk, np.where(d.Diabetes_binary == 1, "sí", "no"), d.Cluster.map(CLUSTER_NOMBRE)]
                ht = "<b>%{customdata[5]}</b><br>IMC %{customdata[0]} · %{customdata[1]} años<br>salud %{customdata[2]}/5 · %{customdata[3]} condiciones<br>diabetes: %{customdata[4]}<extra></extra>"
                if dim == "3D":
                    fig.add_trace(go.Scatter3d(x=d.PC1, y=d.PC2, z=d.PC3, mode="markers", name=nom, customdata=cd, hovertemplate=ht,
                                               marker=dict(size=2.3, color=col, opacity=.75)))
                else:
                    fig.add_trace(go.Scattergl(x=d.PC1, y=d.PC2, mode="markers", name=nom, customdata=cd, hovertemplate=ht,
                                               marker=dict(size=4.5, color=col, opacity=.5, line=dict(width=0))))
            centros = clu["pca"].transform(clu["kmeans"].cluster_centers_)
            inv = {v: k for k, v in clu["mapa"].items()}
            cx = np.array([centros[inv[k]] for k in (1, 2, 3)])
            if dim == "3D":
                fig.add_trace(go.Scatter3d(x=cx[:, 0], y=cx[:, 1], z=cx[:, 2], mode="markers+text", text=[f"C{k}" for k in (1, 2, 3)], name="centroides",
                                           textfont=dict(size=15, color="#fff", family="Sora"), textposition="top center",
                                           marker=dict(size=9, color="#fff", symbol="diamond", line=dict(color="#0a0f26", width=2))))
                ax = lambda i: dict(title=dict(text=f"CP{i + 1} ({es(100 * var[i], 0)} %)", font=dict(color=TEXTO_2)), backgroundcolor="rgba(0,0,0,0)",
                                    gridcolor="rgba(163,174,208,.15)", zerolinecolor="rgba(163,174,208,.25)", tickfont=dict(color=TEXTO_3, size=10), showspikes=False)
                ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.45 * np.cos(t), y=1.45 * np.sin(t), z=.6)))) for t in ang]
                fig.frames = frames
                fig.update_layout(scene=dict(xaxis=ax(0), yaxis=ax(1), zaxis=ax(2), aspectmode="cube", bgcolor="rgba(0,0,0,0)"),
                                  scene_camera=dict(eye=dict(x=1.45, y=0, z=.6)),
                                  updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=1.02, xanchor="left", yanchor="bottom",
                                                    bgcolor="rgba(34,211,238,.15)", bordercolor=CIAN, font=dict(color=TEXTO),
                                                    buttons=[dict(label="⟳  Girar", method="animate",
                                                                  args=[None, dict(frame=dict(duration=90, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                                                             dict(label="❚❚", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])])
                estilo(fig, 640, leyenda=True)
                fig.update_layout(legend=dict(x=.2, y=1.02), margin=dict(l=0, r=0, t=40, b=0))
            else:
                for k, (x0, y0) in zip((1, 2, 3), cx[:, :2]):
                    fig.add_annotation(x=x0, y=y0, text=f"<b>C{k}</b>", showarrow=False, font=dict(size=15, color="#fff", family="Sora"),
                                       bgcolor=C_CLUSTER[k], borderpad=4, opacity=.95)
                fig.update_xaxes(title=f"CP1: carga de mala salud ({es(100 * var[0], 0)} %)")
                fig.update_yaxes(title=f"CP2: edad frente a nivel socioeconómico ({es(100 * var[1], 0)} %)")
                estilo(fig, 560, leyenda=True)
            grafico(fig, key=f"mapa{dim}{color}")
    nota(f"Las tres componentes resumen el {es(100 * sum(var), 0)} % de la variación de las ocho variables. La silueta de {es(R['kmeans']['silueta'], 2)} indica "
         "que la población es un continuo: los perfiles son cortes útiles de ese continuo, no islas separadas.")

    medias = base.groupby("Cluster")[CL_VARS].mean()
    z = (medias - base[CL_VARS].mean()) / base[CL_VARS].std()
    seccion("Perfiles numéricos promedio", "Tres vistas de lo mismo: el radar compara la forma de cada perfil, el mapa muestra cuánto se alejan del promedio y las barras las condiciones clínicas.")
    t1, t2, t3 = st.tabs(["Radar", "Mapa de perfiles", "Condiciones clínicas"])
    with t1:
        with panel("radar"):
            mn, mx = medias.min(), medias.max()
            norm = (medias - mn) / (mx - mn).replace(0, 1)
            fig = go.Figure()
            th = [ET[v] for v in CL_VARS] + [ET[CL_VARS[0]]]
            for k in (1, 2, 3):
                r = list(norm.loc[k]) + [norm.loc[k].iloc[0]]
                fig.add_trace(go.Scatterpolar(r=r, theta=th, name=f"C{k} · {CLUSTER_NOMBRE[k]}", fill="toself", line=dict(color=C_CLUSTER[k], width=2.5),
                                              fillcolor=C_CLUSTER[k], opacity=.38, customdata=list(medias.loc[k]) + [medias.loc[k].iloc[0]],
                                              hovertemplate="%{theta}: %{customdata:.2f}<extra>" + CLUSTER_NOMBRE[k] + "</extra>"))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False, gridcolor="rgba(163,174,208,.15)"),
                                         angularaxis=dict(gridcolor="rgba(163,174,208,.15)", tickfont=dict(size=12.5, color=TEXTO))))
            estilo(fig, 500, leyenda=True)
            fig.update_layout(legend=dict(orientation="h", y=-.06, yanchor="top", x=.5, xanchor="center"), margin=dict(l=80, r=80, t=30, b=50))
            grafico(fig, key="radar")
    with t2:
        with panel("mapaz"):
            fig = go.Figure(go.Heatmap(z=z.T.values, x=[f"C{k} · {CLUSTER_NOMBRE[k]}" for k in z.index], y=[ET[v] for v in CL_VARS], colorscale=DIVERGENTE,
                                       zmid=0, zmin=-1.6, zmax=1.6, xgap=4, ygap=4, text=[[es(medias.loc[k, v], 1) for k in z.index] for v in CL_VARS],
                                       texttemplate="%{text}", textfont=dict(size=13, color="#fff"),
                                       colorbar=dict(title=dict(text="desv. est.", font=dict(color=TEXTO_2, size=11)), tickfont=dict(color=TEXTO_2), thickness=10),
                                       hovertemplate="%{x}<br>%{y}: %{text}<br>%{z:+.2f} desv. est. del promedio<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showgrid=False); fig.update_xaxes(side="top")
            grafico(estilo(fig, 460), key="mapaz")
    with t3:
        with panel("cond"):
            ext = base.groupby("Cluster")[["HighBP", "HighChol", "PhysActivity", "DiffWalk", "Smoker", "Diabetes_binary"]].mean() * 100
            nom = ["Hipertensión", "Colesterol alto", "Hace ejercicio", "Le cuesta caminar", "Ha fumado", "Diabetes (validación)"]
            fig = go.Figure([go.Bar(x=nom, y=ext.loc[k], name=f"C{k} · {CLUSTER_NOMBRE[k]}", marker=dict(color=C_CLUSTER[k], cornerradius=5),
                                    text=[es(v, 0) + "%" for v in ext.loc[k]], textposition="outside", textfont=dict(size=11, color=TEXTO),
                                    hovertemplate=CLUSTER_NOMBRE[k] + "<br>%{x}: %{y:.1f} %<extra></extra>") for k in (1, 2, 3)])
            fig.update_layout(barmode="group", bargap=.22, bargroupgap=.06); fig.update_yaxes(range=[0, 100], ticksuffix=" %")
            grafico(estilo(fig, 420, leyenda=True), key="cond")
    nota("El perfil de riesgo cardiometabólico es el paciente silencioso: casi siete de cada diez con hipertensión y seis con colesterol alto, pero pocos días de malestar. "
         "El perfil de salud deteriorada suma carga física, emocional y económica: necesita atención integral, no solo control de la glucosa.")

    seccion("¿Por qué k = 3?", "Método del codo, coeficiente de silueta y contraste con otros dos algoritmos.")
    tk = pd.DataFrame(R["kmeans_k"])
    a, b = st.columns([1.3, 1], gap="medium")
    with a:
        with panel("codo"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tk.k, y=tk.Inercia / 1e3, name="inercia (miles)", mode="lines+markers", line=dict(color=VIOLETA, width=3, shape="spline"),
                                     marker=dict(size=8), yaxis="y", hovertemplate="k = %{x}<br>inercia %{y:,.0f} mil<extra></extra>"))
            fig.add_trace(go.Scatter(x=tk.k, y=tk.Silueta, name="silueta", mode="lines+markers", line=dict(color=CIAN, width=3, shape="spline"),
                                     marker=dict(size=[16 if k == 3 else 8 for k in tk.k], color=[MAGENTA if k == 3 else CIAN for k in tk.k]), xaxis="x2", yaxis="y2",
                                     hovertemplate="k = %{x}<br>silueta %{y:.3f}<extra></extra>"))
            fig.update_layout(grid=dict(rows=1, columns=2, pattern="independent"), xaxis=dict(title="k", dtick=1, domain=[0, .46]),
                              xaxis2=dict(title="k", dtick=1, domain=[.56, 1]), yaxis=dict(title="inercia (miles)"), yaxis2=dict(title="silueta", anchor="x2"))
            fig.add_annotation(x=3, y=float(tk.set_index("k").Silueta[3]), xref="x2", yref="y2", text="k elegido", showarrow=True, arrowcolor=MAGENTA,
                               font=dict(color=MAGENTA), ay=-30)
            grafico(estilo(fig, 300, leyenda=True), key="codo")
    with b:
        j = R["jerarquico"]
        st.markdown(f'<div class="clc" style="--c:{VIOLETA}"><div class="nom">Validación cruzada de algoritmos</div>'
                    f'<p>Con k = 2 la silueta es mayor, pero solo separa a sanos de no sanos. Con k = 3 aparecen dos riesgos distintos, y esa diferencia es la que sirve para actuar.</p>'
                    f'<p><b style="color:#fff">Ward jerárquico:</b> mismo orden de riesgo, índice de Rand ajustado de {es(j["ari"], 2)}.</p>'
                    f'<p><b style="color:#fff">DBSCAN:</b> no útil con datos discretos de densidad casi uniforme.</p>'
                    f'<p><b style="color:#fff">V de Cramér clúster–diabetes:</b> {es(R["kmeans"]["v_cramer"], 2)} (χ², p &lt; 0,001).</p></div>', unsafe_allow_html=True)
    fuente("Fuente: CDC, BRFSS 2023. K-Means, escalador y PCA ajustados con el conjunto de entrenamiento de la Fase 3.")
