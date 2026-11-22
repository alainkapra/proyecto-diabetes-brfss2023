"""Reentrena los modelos finales de la Fase 3 y genera los artefactos del dashboard.

Uso (desde la raíz del repositorio):
    python -m src.entrenar_modelos

Reproduce exactamente la partición 70/15/15 estratificada de la Fase 2 (semilla 42) y
reentrena, solo con el conjunto de entrenamiento, los modelos finales con los
hiperparámetros que la búsqueda con validación cruzada de la Fase 3 eligió:

  * XGBoost de clasificación (modelo desplegado en el módulo de predicción en vivo).
  * XGBoost de regresión del IMC.
  * K-Means (k = 3) sobre ocho variables estandarizadas, más un PCA de 3 componentes.

Salidas:
  models/modelo_clasificacion_xgboost.joblib
  models/modelo_regresion_xgboost.joblib
  models/modelo_clustering_kmeans.joblib
  models/metadatos_modelos.json
  data/processed/brfss2023_dashboard.parquet   (datos + clúster + partición)
  data/processed/muestra_pca.parquet           (muestra para el gráfico de dispersión)
  data/processed/predicciones_prueba.parquet   (probabilidades del conjunto de prueba)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.preprocesamiento import (CL_VARS, ENTRADAS, RANDOM_STATE, TARGET_CLF, Winsorizer, pipe_arbol_clf,
                                  pipe_reg_arbol, vars_cluster, vars_arbol, cardio_risk, categoria_imc)

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "raw" / "cdc_brfss_diabetes_2023.csv"
MODELOS = RAIZ / "models"
PROC = RAIZ / "data" / "processed"

# Hiperparámetros elegidos por RandomizedSearchCV (Fase 3, sección 5)
XGB_CLF_OPT = dict(colsample_bytree=0.8, learning_rate=0.03, max_depth=5, min_child_weight=1,
                   n_estimators=400, reg_lambda=10, subsample=0.7)
XGB_REG_OPT = dict(colsample_bytree=0.6, learning_rate=0.03, max_depth=6, min_child_weight=1,
                   n_estimators=400, reg_lambda=5, subsample=0.85)


def umbral_f1(y, p) -> float:
    """Umbral que maximiza el F1 de la clase positiva (se elige en validación, nunca en prueba)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.argmax(f1[:-1])])


def main() -> None:
    t0 = time.time()
    MODELOS.mkdir(exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATOS).astype("int16")
    X_all, y_all = df[ENTRADAS], df[TARGET_CLF]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_all, y_all, test_size=0.30, stratify=y_all, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    lo, hi = Winsorizer().fit(X_train).limites_["BMI"]
    print(f"Partición: train {len(X_train):,} · val {len(X_val):,} · test {len(X_test):,} · razón de desbalance {ratio:.2f}")

    # 1. Clasificación: XGBoost optimizado
    base = dict(scale_pos_weight=ratio, tree_method="hist", eval_metric="aucpr", n_jobs=2, random_state=RANDOM_STATE)
    clf = pipe_arbol_clf(XGBClassifier(**base, **XGB_CLF_OPT)).fit(X_train, y_train)
    p_val, p_test = clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]
    umbral = umbral_f1(y_val, p_val)
    pred = (p_test >= umbral).astype(int)
    met_clf = {"auc_roc": roc_auc_score(y_test, p_test), "auc_pr": average_precision_score(y_test, p_test),
               "f1": f1_score(y_test, pred), "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred),
               "recall_05": recall_score(y_test, (p_test >= 0.5).astype(int)),
               "precision_05": precision_score(y_test, (p_test >= 0.5).astype(int)),
               "accuracy": float((pred == y_test.values).mean())}
    print("Clasificación (prueba):", {k: round(v, 4) for k, v in met_clf.items()}, f"umbral {umbral:.4f}")

    # 2. Regresión del IMC: XGBoost optimizado (entrena con el IMC recortado, se evalúa contra el original)
    R = df.drop(columns=["BMI"])
    Rtr, Rte = R.loc[X_train.index], R.loc[X_test.index]
    reg = pipe_reg_arbol(XGBRegressor(tree_method="hist", n_jobs=2, random_state=RANDOM_STATE, **XGB_REG_OPT))
    reg.fit(Rtr, df.loc[X_train.index, "BMI"].astype(float).clip(lo, hi))
    yhat = reg.predict(Rte); yte = df.loc[X_test.index, "BMI"].astype(float)
    met_reg = {"r2": r2_score(yte, yhat), "rmse": float(np.sqrt(mean_squared_error(yte, yhat))), "mae": mean_absolute_error(yte, yhat)}
    print("Regresión IMC (prueba):", {k: round(v, 4) for k, v in met_reg.items()})

    # 3. Segmentación: K-Means k = 3 + PCA 3D (ajustados solo con entrenamiento)
    Ctr = vars_cluster(X_train, (lo, hi))
    esc = StandardScaler().fit(Ctr)
    Ztr = esc.transform(Ctr)
    km = KMeans(3, n_init=10, random_state=RANDOM_STATE).fit(Ztr)
    orden = np.argsort([-(km.labels_ == c).sum() for c in range(3)])
    mapa = {int(o): i + 1 for i, o in enumerate(orden)}           # C1 = el más grande
    pca = PCA(3, random_state=RANDOM_STATE).fit(Ztr)
    Z_all = esc.transform(vars_cluster(X_all, (lo, hi)))
    clus = pd.Series(km.predict(Z_all)).map(mapa).values
    P_all = pca.transform(Z_all)

    # 4. Datos para el dashboard
    split = pd.Series("train", index=df.index)
    split.loc[X_val.index] = "val"; split.loc[X_test.index] = "test"
    dash = df.copy()
    dash["CardioRisk"] = cardio_risk(dash).astype("int8")
    dash["BMI_cat"] = pd.Categorical(categoria_imc(dash["BMI"]))
    dash["Cluster"] = clus.astype("int8")
    dash["Split"] = pd.Categorical(split)
    dash["Year"] = np.int16(2023)
    dash.to_parquet(PROC / "brfss2023_dashboard.parquet", index=False)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.sort(rng.choice(len(df), 12000, replace=False))
    muestra = dash.iloc[idx][["Diabetes_binary", "Cluster", "BMI", "Age", "GenHlth", "MentHlth", "PhysHlth",
                              "Income", "Education", "CardioRisk", "HighBP", "Sex"]].copy()
    muestra[["PC1", "PC2", "PC3"]] = P_all[idx]
    muestra.to_parquet(PROC / "muestra_pca.parquet", index=False)

    pd.DataFrame({"y": y_test.values.astype("int8"), "proba": p_test.astype("float32")}).to_parquet(PROC / "predicciones_prueba.parquet", index=False)

    # 5. Exportación de modelos y metadatos
    joblib.dump(clf, MODELOS / "modelo_clasificacion_xgboost.joblib")
    joblib.dump(reg, MODELOS / "modelo_regresion_xgboost.joblib")
    joblib.dump({"escalador": esc, "kmeans": km, "mapa": mapa, "pca": pca, "variables": CL_VARS, "limites_imc": (lo, hi)},
                MODELOS / "modelo_clustering_kmeans.joblib")
    meta = {"umbral_f1": umbral, "prevalencia_train": float(y_train.mean()), "razon_desbalance": ratio,
            "limites_imc": [lo, hi], "metricas_prueba_clasificacion": met_clf, "metricas_prueba_regresion": met_reg,
            "pca_varianza": [float(v) for v in pca.explained_variance_ratio_],
            "hiperparametros_clf": XGB_CLF_OPT, "hiperparametros_reg": XGB_REG_OPT,
            "n": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
            "versiones": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    json.dump(meta, open(MODELOS / "metadatos_modelos.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Listo en {time.time() - t0:.0f} s. Artefactos en {MODELOS} y {PROC}")


if __name__ == "__main__":
    main()
