"""Genera las tablas del artículo (tablas/*.tex) a partir de los resultados guardados de las Fases 2 y 3.
Así, cada cifra de las tablas proviene de la ejecución de los cuadernos y no se transcribe a mano."""
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "resultados"
META = Path(__file__).resolve().parents[2] / "models" / "metadatos_modelos.json"
R2 = json.load(open(RES / "resultados_fase2.json"))
R3 = json.loads(open(RES / "resultados_fase3.json").read().replace("NaN", "null"))
M = json.load(open(META))
OUT = Path(__file__).parent / "tablas"


def n(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", "{,}").replace("X", ".")


def w(nombre, filas):
    (OUT / f"{nombre}.tex").write_text("\n".join(" & ".join(map(str, f)) + r" \\" for f in filas) + "\n\\bottomrule\n", encoding="utf-8")


esc = lambda s: s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")

# Imputación simulada
w("imputacion", [[x["Mecanismo"], esc(x["Imputador"]).replace("KNNImputer (k = 5)", "KNNImputer ($k=5$)"), n(x["RMSE"], 2), n(x["Sesgo"], 2)] for x in R2["imputacion"]])
# Atípicos
w("atipicos", [[esc(x["Alternativa"]).replace("Winsorization", "Winsorización").replace("p1–p99", "p1--p99"), n(x["Media"], 2), n(x["Desv. est."], 2),
                n(x["Curtosis"], 2), n(x["% registros eliminados"], 2)] for x in R2["atipicos"]["tabla"]])
# Desbalance
nom = {"Sin tratamiento": "Sin tratamiento", "class_weight = balanced": "Pesos de clase", "SMOTE": "SMOTE",
       "Submuestreo aleatorio": "Submuestreo", "Umbral ajustado (0.22)": "Umbral (0,22)"}
w("desbalance", [[nom[x["Estrategia"]], n(x["Accuracy"]), n(x["Precision"]), n(x["Recall"]), n(x["F1 (clase 1)"]), n(x["AUC-PR"])] for x in R2["desbalance"]])
# Modelos base (validación)
w("base", [[esc(x["Modelo"]).replace(" (baseline)", "").replace("Dummy (clase mayoritaria)", "Dummy mayoritario").replace("Dummy (estratificado)", "Dummy estratificado"),
             n(x["Accuracy"]), n(x["Recall"]), n(x["F1 (clase 1)"]), n(x["AUC-ROC"]), n(x["AUC-PR"])]
            for x in R2["baseline_clf"] if x["Conjunto"] == "Validación"])
# Ablación
abl = {"Solo variables originales (17 predictores)": "Solo originales (17)", "Originales + derivadas (24 tras codificar)": "Con derivadas (24)",
       "Con derivadas, entrenamiento sin duplicados": "Con derivadas, sin duplicados"}
w("ablacion", [[abl[x["Variante"]], n(x["Recall"]), n(x["F1 (clase 1)"]), n(x["AUC-ROC"]), n(x["AUC-PR"])] for x in R2["ablacion"]])
# Clasificadores iniciales (train/val)
w("clf_inicial", [[esc(x["Modelo"]).replace(" (baseline Fase 2)", ""), "Entr." if x["Conjunto"] == "Entrenamiento" else "Val.", n(x["F1"]), n(x["AUC-ROC"]), n(x["AUC-PR"]), n(x["Tiempo (s)"], 1)]
                   for x in R3["clf_act5"]])
# Hiperparámetros
w("hiper", [[esc(x["Modelo"]).replace("XGBoost (clasificación)", "XGB clasif.").replace("Random Forest (clasificación)", "RF clasif.").replace("XGBoost (regresión)", "XGB regres."),
              x["Métrica (CV)"], n(x["Inicial"], 4), n(x["Optimizado"], 4), n(x["Desv. entre pliegues"], 4), n(x["Tiempo (s)"], 0)] for x in R3["hiperparametros"]])
mejores = {x["Modelo"]: x["Mejores hiperparámetros"] for x in R3["hiperparametros"]}
(OUT / "mejores.tex").write_text("\n".join(
    rf"\item \textbf{{{esc(k)}}}: \texttt{{{esc(v).replace('.', '{,}')}}}" for k, v in mejores.items()) + "\n", encoding="utf-8")
# Prueba (clasificación)
w("prueba", [[esc(x["Modelo"]).replace(" (optimizado)", "").replace(" (baseline)", "").replace("Regresión logística", "Reg. logística").replace("MLP (PyTorch)", "MLP"),
               n(x["Umbral"], 2), n(x["Precision"]), n(x["Recall"]), n(x["F1"]), n(x["AUC-ROC"]), n(x["AUC-PR"])] for x in R3["test_clf"]])
# Regresión (validación)
w("regresion", [[esc(x["Modelo"]).replace(" (baseline Fase 2)", "").replace("XGBoost (objetivo logarítmico)", "XGBoost (log)"), n(x["R² (train)"]), n(x["R² (val)"]),
                  n(x["RMSE (val)"], 2), n(x["MAE (val)"], 2)] for x in R3["reg_act5"]])
# DBSCAN
w("dbscan", [[n(x["eps"], 1), x["Clústeres"], n(x["Ruido (%)"], 1), n(x["Mayor clúster (%)"], 1), n(x["Silueta (sin ruido)"])] for x in R3["dbscan"]["tabla"]])
# Perfiles
PF, EX = R3["kmeans"]["perfil"], R3["kmeans"]["extra"]
w("perfiles", [
    ["Población (\\%)"] + [n(x["pct"], 1) for x in PF],
    ["Diabetes (\\%)"] + [n(100 * x["diabetes"], 1) for x in PF],
    ["IMC medio"] + [n(x["BMI"], 1) for x in PF],
    ["Edad (código 1--13)"] + [n(x["Age"], 1) for x in PF],
    ["Salud general (1--5)"] + [n(x["GenHlth"], 1) for x in PF],
    ["Días malestar físico"] + [n(x["PhysHlth"], 1) for x in PF],
    ["Días malestar mental"] + [n(x["MentHlth"], 1) for x in PF],
    ["Hipertensión (\\%)"] + [n(100 * x["HighBP"], 1) for x in EX],
    ["Colesterol alto (\\%)"] + [n(100 * x["HighChol"], 1) for x in EX],
    ["Dificultad caminar (\\%)"] + [n(100 * x["DiffWalk"], 1) for x in EX],
    ["Ingreso (1--7)"] + [n(x["Income"], 1) for x in PF],
])
print("Tablas generadas:", sorted(p.name for p in OUT.glob("*.tex")))
