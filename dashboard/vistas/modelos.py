"""Desempeño del modelo: KPI de prueba, umbral interactivo (retrato de 100 personas, matriz de confusión, ROC y PR),
comparación con intervalos de confianza, SHAP, entrenamiento animado de la red neuronal, regresión del IMC y objetivos."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from utils import (AMBAR, CIAN, LIMA, MAGENTA, TEXTO, TEXTO_2, TEXTO_3, VAR_ES, VIOLETA, cargar_datos, cargar_modelos, cargar_predicciones,
                   cargar_resultados, es, estilo, fuente, grafico, hero, kpis, nota, panel, retrato, seccion, titulo_grafico)
from src.preprocesamiento import ENTRADAS

CORTO = {"Regresión logística (baseline)": "Regresión logística", "Random Forest (optimizado)": "Random Forest",
         "XGBoost (optimizado)": "XGBoost", "MLP (PyTorch)": "Red neuronal"}
C_MOD = {"Regresión logística": TEXTO_3, "Random Forest": VIOLETA, "XGBoost": CIAN, "Red neuronal": AMBAR}


@st.cache_data
def curvas():
    p = cargar_predicciones()
    fpr, tpr, _ = roc_curve(p.y, p.proba)
    pr, rc, _ = precision_recall_curve(p.y, p.proba)
    s1, s2 = max(1, len(fpr) // 600), max(1, len(rc) // 600)
    return (fpr[::s1], tpr[::s1]), (rc[::s2], pr[::s2])


@st.cache_data
def pred_regresion():
    _, reg, _, _ = cargar_modelos()
    d = cargar_datos()
    t = d[d.Split == "test"].sample(6000, random_state=42)
    cols = ["Diabetes_binary"] + [c for c in ENTRADAS if c != "BMI"]
    return pd.DataFrame({"real": t.BMI.values, "pred": reg.predict(t[cols])})


def vista():
    R = cargar_resultados()
    _, _, _, meta = cargar_modelos()
    m = meta["metricas_prueba_clasificacion"]; mr = meta["metricas_prueba_regresion"]
    hero("KPI de la solución · conjunto de prueba", "¿Qué tan bien predice el modelo?",
         "Todas las cifras se midieron en 40.916 personas que ningún modelo vio al entrenarse. Mueva el umbral de decisión para ver, en vivo, "
         "el equilibrio entre detectar casos y generar falsas alarmas.")
    kpis([dict(etiqueta="AUC-ROC", valor=m["auc_roc"], dec=3, icono="🎯", color=CIAN, barra=100 * m["auc_roc"], nota="capacidad de ordenar el riesgo"),
          dict(etiqueta="AUC-PR", valor=m["auc_pr"], dec=3, icono="📈", color=VIOLETA, barra=100 * m["auc_pr"], nota="2,8 veces la línea base"),
          dict(etiqueta="F1 con umbral óptimo", valor=m["f1"], dec=3, icono="⚖️", color=AMBAR, barra=100 * m["f1"], nota=f"umbral {es(meta['umbral_f1'], 2)} (validación)"),
          dict(etiqueta="R² del IMC", valor=mr["r2"], dec=3, icono="📏", color=LIMA, barra=100 * mr["r2"] / .3, nota=f"MAE {es(mr['mae'], 2)} kg/m²"),
          dict(etiqueta="Silueta K-Means", valor=R["kmeans"]["silueta"], dec=3, icono="🧩", color=MAGENTA, barra=100 * R["kmeans"]["silueta"] / .3, nota="k = 3 perfiles")], alto=168)

    seccion("Umbral de decisión en vivo", "Por encima del umbral, la persona se envía a una prueba de laboratorio. Cada punto representa a un adulto de cada 100 tamizados.")
    p = cargar_predicciones()
    u = st.slider("Umbral de decisión", 0.05, 0.95, 0.50, 0.01, help=f"0,50 = tamizaje sensible · {es(meta['umbral_f1'], 2)} = máximo F1 en validación")
    pred = (p.proba >= u).astype(int); y = p.y
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    N = len(y); n_tp, n_fn, n_fp = round(tp * 100 / N), round(fn * 100 / N), round(fp * 100 / N)
    rec = tp / (tp + fn); prec = tp / max(tp + fp, 1); esp = tn / (tn + fp); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    kpis([dict(etiqueta="Sensibilidad", valor=100 * rec, dec=1, sufijo="%", icono="🔎", color=MAGENTA, barra=100 * rec, nota="diabéticos detectados"),
          dict(etiqueta="Precisión (VPP)", valor=100 * prec, dec=1, sufijo="%", icono="🧪", color=AMBAR, barra=100 * prec, nota="señalados que tienen diabetes"),
          dict(etiqueta="Especificidad", valor=100 * esp, dec=1, sufijo="%", icono="🛡️", color=LIMA, barra=100 * esp, nota="sanos descartados"),
          dict(etiqueta="F1", valor=f1, dec=3, icono="⚖️", color=CIAN, barra=100 * f1, nota=f"exactitud {es(100 * (tp + tn) / N, 1)} %")], alto=160)
    a, b = st.columns([1.15, 1], gap="medium")
    with a:
        with panel("retrato"):
            titulo_grafico(f"Si se tamizara a 100 adultos con umbral {es(u, 2)}", f"{n_tp + n_fp} irían a prueba · {n_tp} tienen diabetes · {n_fn} casos sin detectar")
            st.markdown(retrato([(n_tp, MAGENTA, MAGENTA), (n_fn, "none", MAGENTA), (n_fp, AMBAR, AMBAR)], total=100, cols=20, r=7, paso=18.5)
                        + f'<div class="leyenda-retrato"><span><i style="background:{MAGENTA}"></i>diabetes detectada</span>'
                          f'<span><i style="border:1.6px solid {MAGENTA}"></i>diabetes no detectada</span>'
                          f'<span><i style="background:{AMBAR}"></i>falsa alarma</span><span><i style="background:rgba(163,174,208,.25)"></i>sano descartado</span></div>',
                        unsafe_allow_html=True)
    with b:
        with panel("cm"):
            titulo_grafico("Matriz de confusión", "Conteos sobre las 40.916 personas de prueba")
            z = np.array([[tn, fp], [fn, tp]])
            txt = [[f"<b>{es(v, 0)}</b><br>{es(100 * v / N, 1)} %" for v in fila] for fila in z]
            fig = go.Figure(go.Heatmap(z=[[tn / (tn + fp), fp / (tn + fp)], [fn / (fn + tp), tp / (fn + tp)]], x=["predice: sin riesgo", "predice: con riesgo"],
                                       y=["real: sin diabetes", "real: con diabetes"], colorscale=[[0, "#141a3d"], [.5, "#5b2f8f"], [1, MAGENTA]], showscale=False,
                                       xgap=5, ygap=5, text=txt, texttemplate="%{text}", textfont=dict(size=15, color="#fff"),
                                       hovertemplate="%{y} · %{x}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            grafico(estilo(fig, 300), key="cm")
    (fx, ty), (rx, py) = curvas()
    a, b = st.columns(2, gap="medium")
    with a:
        with panel("roc"):
            titulo_grafico(f"Curva ROC · área {es(m['auc_roc'], 3)}", "El punto brillante es el umbral elegido")
            fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TEXTO_3, dash="dot", width=1), hoverinfo="skip", showlegend=False),
                             go.Scatter(x=fx, y=ty, mode="lines", line=dict(color=CIAN, width=3), fill="tozeroy", fillcolor="rgba(34,211,238,.10)", showlegend=False,
                                        hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"),
                             go.Scatter(x=[1 - esp], y=[rec], mode="markers", showlegend=False, marker=dict(size=16, color=MAGENTA, line=dict(color="#fff", width=2)))])
            fig.update_xaxes(title="tasa de falsos positivos", range=[0, 1]); fig.update_yaxes(title="sensibilidad", range=[0, 1.02])
            grafico(estilo(fig, 320), key="roc")
    with b:
        with panel("pr"):
            titulo_grafico(f"Curva precisión–sensibilidad · área {es(m['auc_pr'], 3)}", f"La línea punteada es el azar ({es(100 * y.mean(), 1)} %)")
            fig = go.Figure([go.Scatter(x=rx, y=py, mode="lines", line=dict(color=VIOLETA, width=3), fill="tozeroy", fillcolor="rgba(167,139,250,.10)", showlegend=False,
                                        hovertemplate="recall %{x:.2f} · precisión %{y:.2f}<extra></extra>"),
                             go.Scatter(x=[rec], y=[prec], mode="markers", showlegend=False, marker=dict(size=16, color=MAGENTA, line=dict(color="#fff", width=2)))])
            fig.add_hline(y=float(y.mean()), line=dict(color=TEXTO_3, dash="dot", width=1))
            fig.update_xaxes(title="sensibilidad", range=[0, 1]); fig.update_yaxes(title="precisión", range=[0, 1.02])
            grafico(estilo(fig, 320), key="pr")

    seccion("Cuatro modelos, un mismo techo", "Desempeño en prueba con intervalo de confianza del 95 % (bootstrap de 400 remuestreos).")
    t = pd.DataFrame(R["test_clf"]); t = t[t.Umbral == 0.5]; t["M"] = t.Modelo.map(CORTO); bs = R["bootstrap"]
    a, b = st.columns(2, gap="medium")
    for col, met, rango in ((a, "AUC-ROC", [.80, .825]), (b, "AUC-PR", [.38, .44])):
        with col:
            with panel(met):
                titulo_grafico(met)
                orden = t.sort_values(met)
                fig = go.Figure()
                for _, r in orden.iterrows():
                    lo, hi = bs[r.Modelo][met]; c_ = C_MOD[r.M]
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[r.M, r.M], mode="lines", line=dict(color=c_, width=6), opacity=.45, hoverinfo="skip", showlegend=False))
                    fig.add_trace(go.Scatter(x=[r[met]], y=[r.M], mode="markers+text", marker=dict(size=15, color=c_, line=dict(color="#fff", width=1.5)),
                                             text=[es(r[met], 3)], textposition="top center", textfont=dict(size=12, color=TEXTO), cliponaxis=False, showlegend=False,
                                             hovertemplate=f"{r.M}<br>{met} %{{x:.3f}} (IC {es(lo, 3)} a {es(hi, 3)})<extra></extra>"))
                fig.update_xaxes(range=rango); fig.update_yaxes(showgrid=False, tickfont=dict(size=12.5, color=TEXTO))
                grafico(estilo(fig, 230), key=f"cmp{met}")
    nota("Las diferencias son de milésimas: el límite lo pone la información de la encuesta, no el algoritmo. Se desplegó XGBoost por su mejor AUC-PR y porque "
         "permite explicar cada predicción con valores SHAP exactos.")

    a, b = st.columns([1.1, 1], gap="medium")
    with a:
        seccion("Qué mira el modelo", "Impacto medio de cada variable (valores SHAP). Magenta: sube el riesgo al aumentar; verde: lo baja.")
        with panel("shap"):
            sh = pd.Series(R["shap"]["importancia"]).sort_values().iloc[-12:]
            dr = pd.Series(R["shap"]["direccion"]).reindex(sh.index).fillna(0)
            colr = [MAGENTA if d > .3 else LIMA if d < -.3 else VIOLETA for d in dr]
            fig = go.Figure(go.Bar(x=sh.values, y=[VAR_ES.get(v, v) for v in sh.index], orientation="h", marker=dict(color=colr, cornerradius=5),
                                   text=[es(v, 2) for v in sh.values], textposition="outside", textfont=dict(size=11, color=TEXTO),
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_xaxes(range=[0, sh.max() * 1.18], title="|SHAP| medio (log-odds)"); fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=TEXTO))
            grafico(estilo(fig, 440), key="shap")
    with b:
        h = R["mlp"]
        seccion("La red neuronal aprendiendo", f"Arquitectura {h['arquitectura']} en PyTorch. Pulse ▶ para reproducir el entrenamiento época por época.")
        with panel("mlp"):
            tr_, va_ = h["hist"]["train"], h["hist"]["val"]; E = len(tr_); ep = list(range(1, E + 1))
            def trazas(n):
                return [go.Scatter(x=ep[:n], y=tr_[:n], mode="lines+markers", name="entrenamiento", line=dict(color=CIAN, width=3), marker=dict(size=6)),
                        go.Scatter(x=ep[:n], y=va_[:n], mode="lines+markers", name="validación", line=dict(color=MAGENTA, width=3), marker=dict(size=6))]
            fig = go.Figure(data=trazas(E), frames=[go.Frame(data=trazas(n), name=str(n)) for n in range(1, E + 1)])
            fig.add_vline(x=h["mejor_epoca"], line=dict(color=AMBAR, dash="dot"), annotation_text=f"mejor época: {h['mejor_epoca']}", annotation_font=dict(color=AMBAR))
            fig.update_layout(updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=1.13, xanchor="left", yanchor="bottom",
                                                bgcolor="rgba(34,211,238,.15)", bordercolor=CIAN, font=dict(color=TEXTO),
                                                buttons=[dict(label="▶  Entrenar", method="animate", args=[None, dict(frame=dict(duration=160, redraw=False), transition=dict(duration=120), fromcurrent=False, mode="immediate")])])])
            fig.update_xaxes(title="época", range=[.5, E + .5]); fig.update_yaxes(title="pérdida BCE ponderada", range=[min(tr_ + va_) - .004, max(tr_ + va_) + .004])
            estilo(fig, 380, leyenda=True); fig.update_layout(legend=dict(x=.35, y=1.05))
            grafico(fig, key="mlp")

    seccion("Regresión del IMC", f"XGBoost alcanzó un R² de {es(mr['r2'], 3)}: sin dieta ni actividad física medida, las variables explican menos de una quinta parte de la variación del peso.")
    a, b = st.columns([1, 1.2], gap="medium")
    with a:
        with panel("r2"):
            tr = pd.DataFrame(R["test_reg"]); tt = tr[tr.Conjunto == "test"]
            fig = go.Figure(go.Bar(x=tt["R²"], y=tt.Modelo, orientation="h", marker=dict(color=[TEXTO_3, VIOLETA, CIAN, MAGENTA][:len(tt)], cornerradius=5),
                                   text=[es(v, 3) for v in tt["R²"]], textposition="outside", textfont=dict(color=TEXTO),
                                   customdata=np.c_[tt.RMSE, tt.MAE], hovertemplate="%{y}<br>R² %{x:.3f}<br>RMSE %{customdata[0]:.2f} · MAE %{customdata[1]:.2f}<extra></extra>"))
            fig.add_vline(x=.25, line=dict(color=AMBAR, dash="dot"), annotation_text="meta Fase 1", annotation_font=dict(color=AMBAR))
            fig.update_xaxes(range=[0, .3], title="R² en prueba"); fig.update_yaxes(showgrid=False, tickfont=dict(color=TEXTO))
            grafico(estilo(fig, 300), key="r2")
    with b:
        with panel("dens"):
            d = pred_regresion()
            fig = go.Figure(go.Histogram2d(x=d.real.clip(12, 60), y=d.pred, nbinsx=40, nbinsy=30, showscale=False,
                                           colorscale=[[0, "rgba(0,0,0,0)"], [.02, "#24246a"], [.4, "#7d339b"], [1, MAGENTA]],
                                           hovertemplate="real %{x} · estimado %{y}: %{z} personas<extra></extra>"))
            fig.add_trace(go.Scatter(x=[18, 42], y=[18, 42], mode="lines", line=dict(color=CIAN, dash="dot", width=1.5), name="predicción perfecta"))
            fig.update_xaxes(title="IMC real (kg/m²)", range=[12, 60]); fig.update_yaxes(title="IMC estimado", range=[18, 42])
            grafico(estilo(fig, 300, leyenda=True), key="dens")

    seccion("Metas del proyecto")
    o = R["objetivos"]
    filas = [("F1 de la clase diabetes", "≥ 0,70", es(o["f1_mejor_umbral"], 3), False, f"máximo posible con este modelo: {es(o['f1_max_prueba'], 3)}"),
             ("R² del IMC", "≥ 0,25", es(o["r2_mejor"], 3), False, "límite de la información disponible"),
             ("AUC-ROC (reformulada)", "≥ 0,80", es(m["auc_roc"], 3), True, "buen ordenamiento del riesgo"),
             ("Sensibilidad de tamizaje (reformulada)", "≥ 0,75", es(m["recall_05"], 3), True, "umbral 0,5"),
             ("R² del IMC (reformulada)", "≥ 0,18", es(mr["r2"], 3), True, "XGBoost optimizado"),
             ("Segmentación interpretable", "perfiles útiles", "3 perfiles", True, "prevalencias de 4 % a 28 %")]
    html = "".join(f'<tr><td style="color:#fff">{a_}</td><td class="num">{b_}</td><td class="num">{c_}</td>'
                   f'<td style="color:{LIMA if ok else MAGENTA};font-weight:700">{"✔ alcanzado" if ok else "✖ no alcanzado"}</td><td>{e_}</td></tr>' for a_, b_, c_, ok, e_ in filas)
    st.markdown(f'<table class="tabla-ed"><thead><tr><th>Objetivo</th><th>Meta</th><th>Resultado</th><th>Estado</th><th>Comentario</th></tr></thead><tbody>{html}</tbody></table>',
                unsafe_allow_html=True)
    fuente("Fuente: CDC, BRFSS 2023. Modelos y métricas de la Fase 3.")
