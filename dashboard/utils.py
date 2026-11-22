"""Utilidades compartidas del dashboard: rutas, carga de datos y modelos, etiquetas, sistema visual y componentes.

Sistema visual «nocturno luminoso»: fondo azul noche con degradado, paneles translúcidos, acentos de color vivo
(cian, magenta, ámbar, violeta) que brillan sobre lo oscuro, contadores animados y gráficos interactivos.
Tipografías Sora (titulares y cifras) y Manrope (texto), servidas desde dashboard/static/fonts.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:                     # los modelos referencian funciones de src.preprocesamiento
    sys.path.insert(0, str(RAIZ))

import joblib  # noqa: E402

from src.preprocesamiento import ENTRADAS, vars_cluster  # noqa: E402

# ───────────────────────────── Paleta ─────────────────────────────
NOCHE = "#070b1f"
NOCHE_2 = "#141042"
PANEL = "rgba(255,255,255,0.045)"
BORDE = "rgba(148,163,255,0.16)"
TEXTO = "#e7ecff"
TEXTO_2 = "#a3aed0"
TEXTO_3 = "#6b7699"
CIAN = "#22d3ee"
MAGENTA = "#f472b6"
AMBAR = "#fbbf24"
VIOLETA = "#a78bfa"
LIMA = "#a3e635"
INDIGO_APAGADO = "#46558a"
C_CLUSTER = {1: CIAN, 2: AMBAR, 3: MAGENTA}
C_DIAB = {0: INDIGO_APAGADO, 1: MAGENTA}
SECUENCIAL = ["#121838", "#24246a", "#4b2a8c", "#7d339b", "#b83f9f", "#e45aa6", "#f9a8d4"]
DIVERGENTE = [[0, CIAN], [0.5, "#161c40"], [1, MAGENTA]]
# alias usados por las vistas
TINTA, TINTA_2, TINTA_3 = TEXTO, TEXTO_2, TEXTO_3
CARMIN, LAVANDA, NIEBLA = MAGENTA, CIAN, INDIGO_APAGADO

# ───────────────────────────── Etiquetas ─────────────────────────────
EDAD = {1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49", 7: "50–54",
        8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"}
SALUD = {1: "Excelente", 2: "Muy buena", 3: "Buena", 4: "Regular", 5: "Mala"}
SEXO = {0: "Mujer", 1: "Hombre"}
EDUC = {2: "Hasta secundaria incompleta", 4: "Secundaria completa", 5: "Técnica o universitaria incompleta",
        6: "Universitaria completa"}
INGRESO = {i: f"Nivel {i}" for i in range(1, 8)}
INGRESO[1] = "Nivel 1 (más bajo)"; INGRESO[7] = "Nivel 7 (más alto)"
CAT_IMC = ["Bajo peso", "Normal", "Sobrepeso", "Obesidad I", "Obesidad II", "Obesidad III"]
CLUSTER_NOMBRE = {1: "Jóvenes sanos", 2: "Riesgo cardiometabólico", 3: "Salud deteriorada"}
CLUSTER_CORTO = CLUSTER_NOMBRE
CLUSTER_DESC = {
    1: "Adultos de menor edad y peso, que califican bien su salud, con mayor ingreso y casi sin hipertensión ni colesterol alto.",
    2: "Adultos mayores con hipertensión y colesterol alto frecuentes que, sin embargo, se sienten bien y consultan poco.",
    3: "Personas con muchos días de malestar físico y mental, dificultad para caminar y el ingreso más bajo.",
}
VAR_ES = {
    "Diabetes_binary": "Diabetes", "HighBP": "Hipertensión", "HighChol": "Colesterol alto", "CholCheck": "Control de colesterol (5 años)",
    "BMI": "Índice de masa corporal", "Smoker": "Fumador (≥100 cigarrillos)", "Stroke": "Accidente cerebrovascular",
    "HeartDiseaseorAttack": "Enfermedad coronaria o infarto", "PhysActivity": "Actividad física (30 días)",
    "NoDocbcCost": "No consultó por costo", "GenHlth": "Salud general percibida", "MentHlth": "Días de mala salud mental",
    "PhysHlth": "Días de mala salud física", "DiffWalk": "Dificultad para caminar", "Sex": "Sexo", "Age": "Grupo de edad",
    "Education": "Nivel educativo", "Income": "Nivel de ingreso", "CardioRisk": "Condiciones cardiometabólicas (0–4)",
    "UnhealthyDays": "Días de mala salud (total)", "Age_BMI": "Interacción edad × IMC", "BMI_cat": "Categoría de IMC",
}
BINARIAS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "NoDocbcCost", "DiffWalk"]


def es(x, d=1):
    """Número con coma decimal y punto de miles (formato es-CO)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


# ───────────────────────────── Carga ─────────────────────────────
@st.cache_data(show_spinner="Cargando la encuesta BRFSS 2023…")
def cargar_datos() -> pd.DataFrame:
    """Lee los datos procesados. Si hay archivos de otros años en data/raw (cdc_brfss_diabetes_AAAA.csv),
    se añaden automáticamente para el filtro temporal (solo exploración; los modelos usan 2023)."""
    df = pd.read_parquet(RAIZ / "data" / "processed" / "brfss2023_dashboard.parquet")
    df["BMI_cat"] = pd.Categorical(df["BMI_cat"].astype(str), categories=CAT_IMC, ordered=True)
    extra = []
    for f in sorted((RAIZ / "data" / "raw").glob("cdc_brfss_diabetes_20*.csv")):
        anio = int(f.stem[-4:])
        if anio == 2023:
            continue
        d = pd.read_csv(f).astype("int16")
        d["CardioRisk"] = (d.HighBP + d.HighChol + d.HeartDiseaseorAttack + d.Stroke).astype("int8")
        d["BMI_cat"] = pd.Categorical(pd.cut(d.BMI, [-np.inf, 18.5, 25, 30, 35, 40, np.inf], right=False, labels=CAT_IMC).astype(str),
                                      categories=CAT_IMC, ordered=True)
        d["Cluster"] = np.int8(0); d["Split"] = "otro año"; d["Year"] = np.int16(anio)
        extra.append(d)
    if extra:
        df = pd.concat([df] + extra, ignore_index=True)
    return df


@st.cache_resource(show_spinner="Cargando modelos…")
def cargar_modelos():
    m = RAIZ / "models"
    clf = joblib.load(m / "modelo_clasificacion_xgboost.joblib")
    reg = joblib.load(m / "modelo_regresion_xgboost.joblib")
    clu = joblib.load(m / "modelo_clustering_kmeans.joblib")
    meta = json.load(open(m / "metadatos_modelos.json", encoding="utf-8"))
    return clf, reg, clu, meta


@st.cache_data
def cargar_resultados() -> dict:
    with open(RAIZ / "reports" / "resultados" / "resultados_fase3.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_muestra_pca() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data" / "processed" / "muestra_pca.parquet")


@st.cache_data
def cargar_predicciones() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data" / "processed" / "predicciones_prueba.parquet")


def asignar_cluster(X: pd.DataFrame, clu: dict) -> np.ndarray:
    Z = clu["escalador"].transform(vars_cluster(X[ENTRADAS], clu["limites_imc"]))
    return pd.Series(clu["kmeans"].predict(Z)).map(clu["mapa"]).values


# ───────────────────────────── Filtros ─────────────────────────────
def filtrar(df: pd.DataFrame, s=st.session_state) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if "f_anio" in s:
        m &= df.Year.between(*s.f_anio)
    if s.get("f_sexo"):
        m &= df.Sex.isin([k for k, v in SEXO.items() if v in s.f_sexo])
    if "f_edad" in s:
        inv = {v: k for k, v in EDAD.items()}
        m &= df.Age.between(inv[s.f_edad[0]], inv[s.f_edad[1]])
    if "f_imc" in s:
        m &= df.BMI.between(*s.f_imc)
    if s.get("f_salud"):
        m &= df.GenHlth.isin([k for k, v in SALUD.items() if v in s.f_salud])
    if "f_ingreso" in s:
        m &= df.Income.between(*s.f_ingreso)
    if s.get("f_educ"):
        m &= df.Education.isin([k for k, v in EDUC.items() if v in s.f_educ])
    if "f_cardio" in s:
        m &= df.CardioRisk.between(*s.f_cardio)
    if s.get("f_cluster"):
        m &= df.Cluster.isin([k for k, v in CLUSTER_NOMBRE.items() if v in s.f_cluster])
    if s.get("f_diab", "Todas") != "Todas":
        m &= df.Diabetes_binary == (1 if s.f_diab == "Con diabetes" else 0)
    return df[m]


# ───────────────────────────── Gráficos ─────────────────────────────
FUENTE_UI = "'Manrope', 'Segoe UI', Arial, sans-serif"
FUENTE_TIT = "'Sora', 'Manrope', Arial, sans-serif"


def estilo(fig: go.Figure, alto: int = 360, leyenda: bool = False, **_) -> go.Figure:
    """Estilo nocturno: fondo transparente, rejilla tenue, texto claro, etiquetas emergentes oscuras."""
    fig.update_layout(
        height=alto, margin=dict(l=10, r=16, t=40 if leyenda else 14, b=10, autoexpand=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FUENTE_UI, size=12.5, color=TEXTO_2),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12.5, color=TEXTO_2), itemsizing="constant"),
        hoverlabel=dict(bgcolor="#0d1230", bordercolor=CIAN, font=dict(color=TEXTO, size=12.5, family=FUENTE_UI)),
        separators=",.", bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, automargin=True, linecolor="rgba(163,174,208,.35)", tickfont=dict(color=TEXTO_2, size=12),
                     title_font=dict(color=TEXTO_3, size=12), zeroline=False)
    fig.update_yaxes(gridcolor="rgba(163,174,208,.09)", automargin=True, zeroline=False, showline=False,
                     tickfont=dict(color=TEXTO_2, size=12), title_font=dict(color=TEXTO_3, size=12))
    return fig


CONFIG_PLOTLY = {"displaylogo": False, "modeBarButtonsToRemove": ["autoScale2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2}}


def grafico(fig, **kw):
    return st.plotly_chart(fig, width="stretch", config=CONFIG_PLOTLY, theme=None, **kw)


def panel(clave: str = ""):
    """Panel translúcido (vidrio). La clave debe ser única dentro de la página."""
    st.session_state["_k"] = st.session_state.get("_k", 0) + 1
    return st.container(key=f"panel{clave}{st.session_state['_k']}")


tarjeta = panel


def degradado_barras(n: int, a=CIAN, b=VIOLETA) -> list[str]:
    """Colores interpolados de a hacia b para dar sensación de degradado a una serie de barras."""
    def h2r(h): return [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    ca, cb = h2r(a), h2r(b)
    return [("#%02x%02x%02x" % tuple(int(ca[j] + (cb[j] - ca[j]) * t) for j in range(3))) for t in np.linspace(0, 1, max(n, 1))]


# ───────────────────────────── Componentes de texto ─────────────────────────────
def _nb(t: str) -> str:
    return t.replace(" %", "&nbsp;%").replace(" kg/m²", "&nbsp;kg/m²")


def hero(etiqueta: str, titulo: str, sub: str):
    st.markdown(f'<header class="hero"><span class="hero-tag">{etiqueta}</span><h1>{_nb(titulo)}</h1>'
                f'<p>{_nb(sub)}</p></header>', unsafe_allow_html=True)


portada = lambda titulo, sub, etiqueta="BRFSS 2023": hero(etiqueta, titulo, sub)  # noqa: E731


def seccion(titulo: str, texto: str = ""):
    st.markdown(f'<div class="sec"><h2>{_nb(titulo)}</h2>{f"<p>{_nb(texto)}</p>" if texto else ""}</div>', unsafe_allow_html=True)


def titulo_grafico(titulo: str, sub: str = ""):
    st.markdown(f'<div class="tg"><h3>{_nb(titulo)}</h3>{f"<p>{_nb(sub)}</p>" if sub else ""}</div>', unsafe_allow_html=True)


def fuente(texto: str):
    st.markdown(f'<p class="fuente">{texto}</p>', unsafe_allow_html=True)


def nota(texto: str, tipo: str = "info"):
    st.markdown(f'<aside class="nota nota-{tipo}">{_nb(texto)}</aside>', unsafe_allow_html=True)


def chip_filtros(n: int, total: int):
    if n < total:
        st.markdown(f'<p class="filtro-activo"><span class="pulso"></span>Vista filtrada: {es(n, 0)} de {es(total, 0)} personas '
                    f'({es(100 * n / total, 1)} %)</p>', unsafe_allow_html=True)


# ───────────────────────────── KPI animados (contadores) ─────────────────────────────
@st.cache_resource
def _fuentes_b64() -> str:
    css = ""
    for fam, arch in (("Sora", "sora-latin-wght-normal.woff2"), ("Manrope", "manrope-latin-wght-normal.woff2")):
        b = base64.b64encode((Path(__file__).parent / "static" / "fonts" / arch).read_bytes()).decode()
        css += f"@font-face{{font-family:'{fam}';src:url(data:font/woff2;base64,{b}) format('woff2');font-weight:200 800;}}"
    return css


def kpis(items: list[dict], alto: int = 150, columnas: int | None = None):
    """Tarjetas KPI con contador animado. Cada item: valor (float), dec, sufijo, etiqueta, nota, color, icono."""
    n = columnas or len(items)
    tarjetas = ""
    for i, it in enumerate(items):
        col = it.get("color", CIAN)
        tarjetas += (f'<div class="k" style="--c:{col};animation-delay:{i * 90}ms">'
                     f'<div class="top"><span class="ico">{it.get("icono", "")}</span><span class="lab">{it["etiqueta"]}</span></div>'
                     f'<div class="val"><span class="num" data-v="{it["valor"]}" data-d="{it.get("dec", 0)}">0</span>'
                     f'<span class="suf">{it.get("sufijo", "")}</span></div><div class="nota">{it.get("nota", "")}</div>'
                     f'<div class="barra"><i style="width:{min(100, max(4, it.get("barra", 0)))}%"></i></div></div>')
    html = f"""<style>{_fuentes_b64()}
*{{box-sizing:border-box;margin:0;padding:0}} body{{background:transparent;font-family:'Manrope',sans-serif;color:{TEXTO}}}
.g{{display:grid;grid-template-columns:repeat({n},minmax(0,1fr));gap:14px;padding:4px 2px 8px}}
.k{{position:relative;border-radius:18px;padding:16px 16px 14px;background:linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02));
   border:1px solid rgba(148,163,255,.18);overflow:hidden;opacity:0;transform:translateY(10px);animation:in .7s cubic-bezier(.2,.7,.2,1) forwards}}
.k:before{{content:"";position:absolute;inset:-40% 40% auto -30%;height:120px;background:radial-gradient(closest-side,var(--c),transparent);opacity:.22;filter:blur(8px)}}
.k:hover{{border-color:var(--c);box-shadow:0 0 0 1px var(--c),0 10px 40px -12px var(--c)}}
.top{{display:flex;align-items:center;gap:8px;position:relative}}
.ico{{width:28px;height:28px;border-radius:9px;display:grid;place-items:center;background:rgba(255,255,255,.07);font-size:15px}}
.lab{{font-size:12.5px;color:{TEXTO_2};font-weight:600;letter-spacing:.01em}}
.val{{font-family:'Sora',sans-serif;font-weight:600;font-size:34px;line-height:1.1;margin-top:10px;color:#fff;position:relative;
     text-shadow:0 0 22px var(--c);font-variant-numeric:tabular-nums}}
.suf{{font-size:18px;color:var(--c);margin-left:4px;text-shadow:none}}
.nota{{font-size:12px;color:{TEXTO_2};margin-top:4px;line-height:1.35;position:relative;min-height:16px}}
.barra{{height:4px;border-radius:4px;background:rgba(255,255,255,.08);margin-top:10px;overflow:hidden;position:relative}}
.barra i{{display:block;height:100%;border-radius:4px;background:linear-gradient(90deg,var(--c),#fff);box-shadow:0 0 12px var(--c);
        transform-origin:left;animation:grow 1.4s .3s cubic-bezier(.2,.7,.2,1) both}}
@keyframes in{{to{{opacity:1;transform:none}}}} @keyframes grow{{from{{transform:scaleX(0)}}}}
@media (max-width:720px){{.g{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media (prefers-reduced-motion:reduce){{.k,.barra i{{animation:none;opacity:1;transform:none}}}}
</style><div class="g">{tarjetas}</div>
<script>
const fmt=(v,d)=>v.toLocaleString('es-CO',{{minimumFractionDigits:d,maximumFractionDigits:d}});
document.querySelectorAll('.num').forEach((el,i)=>{{const v=parseFloat(el.dataset.v),d=parseInt(el.dataset.d);const t0=performance.now()+i*90,D=1300;
 const step=t=>{{const p=Math.min(1,Math.max(0,(t-t0)/D)),e=1-Math.pow(1-p,3);el.textContent=fmt(v*e,d);if(p<1)requestAnimationFrame(step)}};
 if(window.matchMedia('(prefers-reduced-motion: reduce)').matches){{el.textContent=fmt(v,d)}}else requestAnimationFrame(step);}});
</script>"""
    components.html(html, height=alto)


def retrato(grupos: list[tuple[int, str, str]], total: int = 100, cols: int = 20, r: float = 6.0, paso: float = 17.0,
            vacio: str = "rgba(163,174,208,.16)", titulo: str = "", animar: bool = True) -> str:
    """Retrato de puntos (icon array) en SVG; los puntos coloreados se encienden en secuencia."""
    filas = int(np.ceil(total / cols))
    w, h = cols * paso, filas * paso
    celdas = []
    for n, relleno, borde in grupos:
        celdas += [(relleno, borde)] * int(n)
    celdas += [(vacio, vacio)] * (total - len(celdas))
    puntos = []
    for k, (f, b) in enumerate(celdas[:total]):
        cx, cy = (k % cols) * paso + paso / 2, (k // cols) * paso + paso / 2
        hueco = f == "none"
        encendido = f != vacio
        anim = f' class="on" style="animation-delay:{k * 12}ms"' if (animar and encendido) else ""
        puntos.append(f'<circle{anim} cx="{cx:.1f}" cy="{cy:.1f}" r="{r - (0.8 if hueco else 0):.1f}" fill="{f}" '
                      f'stroke="{b}" stroke-width="{1.6 if hueco else 0}"/>')
    return (f'<svg class="retrato" viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="{titulo}" '
            f'preserveAspectRatio="xMinYMin meet">{"".join(puntos)}</svg>')


# ───────────────────────────── Hoja de estilos ─────────────────────────────
CSS = f"""
<style>
@font-face {{ font-family:'Sora'; src:url('app/static/fonts/sora-latin-wght-normal.woff2') format('woff2'); font-weight:100 800; font-display:swap; }}
@font-face {{ font-family:'Manrope'; src:url('app/static/fonts/manrope-latin-wght-normal.woff2') format('woff2'); font-weight:200 800; font-display:swap; }}
html, body, .stApp, .stMarkdown, button, input, select, textarea, label, [data-testid="stWidgetLabel"] p {{ font-family:'Manrope', sans-serif; }}
.stApp {{ background:
    radial-gradient(900px 520px at 8% -8%, rgba(34,211,238,.16), transparent 60%),
    radial-gradient(800px 600px at 100% 0%, rgba(167,139,250,.20), transparent 60%),
    radial-gradient(700px 500px at 60% 110%, rgba(244,114,182,.12), transparent 60%),
    linear-gradient(160deg, {NOCHE} 0%, {NOCHE_2} 100%); background-attachment: fixed; color:{TEXTO}; }}
.block-container {{ padding-top: 4.8rem; padding-bottom: 4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: rgba(7,11,31,.72); backdrop-filter: blur(14px); border-bottom:1px solid {BORDE}; }}
[data-testid="stAppDeployButton"] {{ display:none; }}

[data-testid="stTopNavLink"] {{ border-radius:999px !important; padding:.3rem .85rem !important; margin:0 .15rem; transition: background .2s; }}
[data-testid="stTopNavLink"] span {{ color:{TEXTO_2}; font-weight:600; font-size:.9rem; }}
[data-testid="stTopNavLink"]:hover {{ background: rgba(255,255,255,.06) !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background: linear-gradient(90deg, rgba(34,211,238,.22), rgba(167,139,250,.22)) !important;
    box-shadow: inset 0 0 0 1px rgba(34,211,238,.45); }}
[data-testid="stTopNavLink"][aria-current="page"] span {{ color:#fff; }}

[data-testid="stSidebar"] {{ background: linear-gradient(180deg, rgba(10,14,38,.92), rgba(20,16,66,.92)); border-right:1px solid {BORDE}; backdrop-filter: blur(16px); }}
[data-testid="stSidebar"] .lat-t {{ font-family:'Sora'; font-weight:600; font-size:1.15rem; color:#fff; margin:.1rem 0 .2rem; }}
[data-testid="stSidebar"] .lat-s {{ font-size:.84rem; color:{TEXTO_2}; line-height:1.45; margin:0 0 .9rem; }}
.lat-n {{ font-family:'Sora'; font-size:2.1rem; font-weight:600; color:#fff; text-shadow:0 0 22px rgba(34,211,238,.7); line-height:1; }}
.lat-n small {{ display:block; font-family:'Manrope'; font-size:.8rem; color:{TEXTO_2}; text-shadow:none; margin-top:.35rem; font-weight:500; }}
.lat-b {{ height:5px; border-radius:5px; background:rgba(255,255,255,.08); margin:.6rem 0 1.1rem; overflow:hidden; }}
.lat-b i {{ display:block; height:100%; background:linear-gradient(90deg,{CIAN},{VIOLETA},{MAGENTA}); box-shadow:0 0 14px {CIAN}; }}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{ background: linear-gradient(90deg,{CIAN},{VIOLETA}) !important; color:#0a0f26 !important; }}
[data-testid="stSidebar"] .stButton button {{ width:100%; border-radius:999px; border:1px solid rgba(34,211,238,.5); background:transparent; color:{CIAN}; }}
[data-testid="stSidebar"] .stButton button:hover {{ background: {CIAN}; color:#0a0f26; box-shadow:0 0 22px rgba(34,211,238,.6); }}

.hero {{ margin:.2rem 0 1.4rem; animation: sube .8s cubic-bezier(.2,.7,.2,1) both; }}
.hero-tag {{ display:inline-flex; align-items:center; gap:.45rem; font-size:.8rem; font-weight:700; color:{CIAN}; padding:.28rem .75rem;
    border-radius:999px; background:rgba(34,211,238,.1); border:1px solid rgba(34,211,238,.35); }}
.hero-tag:before {{ content:""; width:7px; height:7px; border-radius:50%; background:{CIAN}; box-shadow:0 0 10px {CIAN}; animation: late 1.8s infinite; }}
.hero h1 {{ font-family:'Sora'; font-weight:700; font-size: clamp(2.1rem, 3.6vw, 3.2rem); line-height:1.08; letter-spacing:-.02em; margin:.7rem 0 .6rem; padding:0;
    background: linear-gradient(92deg, #ffffff 0%, #c7f6ff 35%, {CIAN} 60%, {VIOLETA} 80%, {MAGENTA} 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }}
div[data-testid="stMarkdownContainer"] .hero p, .hero p {{ font-size:1.08rem; line-height:1.6; color:{TEXTO_2}; max-width: 54rem; margin:0; }}
@keyframes sube {{ from {{ opacity:0; transform: translateY(14px); }} to {{ opacity:1; transform:none; }} }}
@keyframes late {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.25; }} }}

.sec {{ margin: 2.2rem 0 .8rem; }}
.sec h2 {{ font-family:'Sora'; font-weight:600; font-size:1.5rem; color:#fff; margin:0 0 .3rem; padding:0; letter-spacing:-.01em; }}
div[data-testid="stMarkdownContainer"] .sec p, .sec p {{ color:{TEXTO_2}; font-size:.98rem; margin:0; max-width: 60rem; line-height:1.55; }}
.tg h3 {{ font-family:'Sora'; font-weight:600; font-size:1.05rem; color:#fff; margin:0; padding:0; }}
div[data-testid="stMarkdownContainer"] .tg p, .tg p {{ color:{TEXTO_3}; font-size:.85rem; margin:.2rem 0 0; }}
div[data-testid="stMarkdownContainer"] p.fuente, .fuente {{ font-size:.78rem; color:{TEXTO_3}; margin:.2rem 0 1rem; }}
div[data-testid="stMarkdownContainer"] .nota, .nota {{ position:relative; font-size:.95rem; line-height:1.6; color:{TEXTO}; margin:.6rem 0 1.2rem; padding:.85rem 1.1rem .85rem 1.2rem;
    border-radius:14px; background:linear-gradient(90deg, rgba(34,211,238,.10), rgba(255,255,255,.02)); border:1px solid rgba(34,211,238,.22); }}
.nota:before {{ content:""; position:absolute; left:0; top:12px; bottom:12px; width:3px; border-radius:3px; background:{CIAN}; box-shadow:0 0 12px {CIAN}; }}
.nota-warn {{ background:linear-gradient(90deg, rgba(244,114,182,.12), rgba(255,255,255,.02)) !important; border-color: rgba(244,114,182,.3) !important; }}
.nota-warn:before {{ background:{MAGENTA}; box-shadow:0 0 12px {MAGENTA}; }}
.nota-ok:before {{ background:{LIMA}; box-shadow:0 0 12px {LIMA}; }}
.nota b {{ color:#fff; }}
div[data-testid="stMarkdownContainer"] p.filtro-activo, .filtro-activo {{ display:inline-flex; align-items:center; gap:.5rem; font-size:.85rem; color:{MAGENTA}; margin:-.4rem 0 1rem;
    padding:.3rem .8rem; border-radius:999px; background:rgba(244,114,182,.1); border:1px solid rgba(244,114,182,.3); }}
.pulso {{ width:8px; height:8px; border-radius:50%; background:{MAGENTA}; box-shadow:0 0 10px {MAGENTA}; animation: late 1.4s infinite; }}

div[class*="st-key-panel"] {{ background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.02)); border:1px solid {BORDE};
    border-radius: 20px; padding: 1rem 1.1rem .6rem; backdrop-filter: blur(10px); transition: border-color .25s, box-shadow .25s; }}
div[class*="st-key-panel"]:hover {{ border-color: rgba(34,211,238,.35); box-shadow: 0 18px 50px -28px rgba(34,211,238,.55); }}
div[class*="st-key-ficha"] {{ background: linear-gradient(160deg, rgba(167,139,250,.10), rgba(255,255,255,.02)); border:1px solid rgba(167,139,250,.3);
    border-radius: 20px; padding: 1.2rem 1.3rem .8rem; }}
.ficha-t {{ font-family:'Sora'; font-weight:600; color:#fff; font-size:1rem; margin:.2rem 0 .3rem; }}

.retrato {{ width:100%; height:auto; display:block; filter: drop-shadow(0 0 6px rgba(244,114,182,.35)); }}
.retrato circle.on {{ opacity:0; animation: enc .45s ease forwards; }}
@keyframes enc {{ from {{ opacity:0; transform-box: fill-box; transform-origin:center; transform: scale(.2); }} to {{ opacity:1; transform: scale(1); }} }}
.leyenda-retrato {{ display:flex; flex-wrap:wrap; gap:.4rem 1.2rem; font-size:.85rem; color:{TEXTO_2}; margin:.6rem 0 0; }}
.leyenda-retrato i {{ display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.4rem; vertical-align:-.05rem; }}

.titular {{ font-family:'Sora'; font-weight:600; font-size: clamp(1.5rem, 2.4vw, 2.1rem); line-height:1.2; color:#fff; margin:0 0 .7rem; }}
.titular em {{ font-style:normal; color:{MAGENTA}; text-shadow:0 0 24px rgba(244,114,182,.7); }}
div[data-testid="stMarkdownContainer"] p.cuerpo, .cuerpo {{ color:{TEXTO_2}; font-size:1rem; line-height:1.6; max-width: 38rem; }}

.clc {{ position:relative; border-radius:20px; padding:1.1rem 1.2rem 1rem; height:100%; overflow:hidden;
    background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); border:1px solid var(--c); box-shadow: 0 0 30px -12px var(--c); }}
.clc:after {{ content:""; position:absolute; right:-60px; top:-60px; width:180px; height:180px; border-radius:50%; background: radial-gradient(closest-side, var(--c), transparent); opacity:.28; }}
.clc .nom {{ font-family:'Sora'; font-weight:600; font-size:1.12rem; color:#fff; position:relative; }}
.clc .fila {{ display:flex; gap:1.6rem; margin:.6rem 0 .3rem; position:relative; }}
.clc .big {{ font-family:'Sora'; font-weight:700; font-size:2rem; color:var(--c); text-shadow:0 0 20px var(--c); line-height:1; }}
.clc .lbl {{ font-size:.76rem; color:{TEXTO_2}; margin-top:.25rem; }}
.clc p {{ font-size:.88rem; color:{TEXTO_2}; line-height:1.5; margin:.4rem 0 0; position:relative; }}

.score {{ position:relative; text-align:left; padding:1.2rem 1.4rem; border-radius:22px; border:1px solid var(--c);
    background: radial-gradient(400px 200px at 0% 0%, color-mix(in srgb, var(--c) 30%, transparent), transparent 70%), rgba(255,255,255,.03);
    box-shadow: 0 0 50px -18px var(--c); }}
.score .lbl {{ font-size:.82rem; font-weight:700; color:var(--c); }}
.score .num {{ font-family:'Sora'; font-weight:700; font-size:4.4rem; line-height:1; color:#fff; text-shadow:0 0 30px var(--c); animation: sube .6s both; }}
.score .num small {{ font-size:1.3rem; color:{TEXTO_3}; text-shadow:none; }}
.score p {{ color:{TEXTO_2}; font-size:.95rem; line-height:1.55; margin:.5rem 0 0; }}
.score p b {{ color:#fff; }}

.stack {{ display:flex; flex-wrap:wrap; gap:.5rem; margin:.4rem 0 1rem; }}
.stack span {{ font-size:.85rem; font-weight:600; padding:.35rem .75rem; border-radius:999px; border:1px solid {BORDE}; background:rgba(255,255,255,.04); color:{TEXTO}; }}
.stack span b {{ color:{CIAN}; font-weight:700; margin-left:.3rem; }}
.paso {{ border-radius:16px; padding:.9rem 1rem; height:100%; background:rgba(255,255,255,.04); border:1px solid {BORDE}; }}
.paso .n {{ font-family:'Sora'; font-weight:700; font-size:1.5rem; background:linear-gradient(90deg,{CIAN},{VIOLETA}); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.paso b {{ display:block; color:#fff; font-size:1rem; margin:.25rem 0 .2rem; }}
.paso span {{ font-size:.86rem; color:{TEXTO_2}; line-height:1.45; }}
.tabla-ed {{ width:100%; border-collapse:collapse; font-size:.9rem; margin:.3rem 0 1rem; }}
.tabla-ed th {{ text-align:left; color:{CIAN}; font-weight:700; border-bottom:1px solid rgba(34,211,238,.4); padding:.5rem .6rem .5rem 0; }}
.tabla-ed td {{ border-bottom:1px solid rgba(163,174,208,.12); padding:.5rem .6rem .5rem 0; color:{TEXTO_2}; vertical-align:top; }}
.tabla-ed td.num {{ color:#fff; font-variant-numeric: tabular-nums; }}

[role="tablist"] {{ gap:.4rem; border-bottom:none; background:rgba(255,255,255,.04); padding:.3rem; border-radius:999px; width:fit-content; max-width:100%; border:1px solid {BORDE}; }}
[role="tab"] {{ border-radius:999px !important; padding:.35rem .95rem !important; }}
[role="tab"] p {{ color:{TEXTO_2}; font-weight:600; font-size:.9rem; }}
[role="tab"][aria-selected="true"] {{ background: linear-gradient(90deg, rgba(34,211,238,.25), rgba(167,139,250,.25)) !important; }}
[role="tab"][aria-selected="true"] p {{ color:#fff; }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display:none; }}
.stButton button {{ border-radius:999px; }}
.pie {{ color:{TEXTO_3}; font-size:.8rem; margin-top:3rem; padding-top:1rem; border-top:1px solid {BORDE}; line-height:1.5; }}
@media (prefers-reduced-motion: reduce) {{ .hero, .retrato circle.on, .score .num {{ animation:none !important; opacity:1 !important; }} }}
</style>
"""
