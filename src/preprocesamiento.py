"""Transformaciones de preprocesamiento del proyecto (Fases 2 y 3).

Este módulo reúne, en un solo lugar importable, las funciones y clases que los
cuadernos definen en línea. Los modelos guardados en /models referencian estas
funciones, por lo que el dashboard y los scripts las importan desde aquí.

Todas las transformaciones aprenden sus parámetros solo con el conjunto de
entrenamiento (dentro de un Pipeline), lo que evita la fuga de datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_CLF = "Diabetes_binary"
TARGET_REG = "BMI"

BIN = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
       "PhysActivity", "NoDocbcCost", "DiffWalk", "Sex"]
ORD = ["GenHlth", "Age", "Education", "Income"]
ORD_CATS = [list(range(1, 6)), list(range(1, 14)), list(range(1, 7)), list(range(1, 12))]
CAT_IMC = ["Bajo peso", "Normal", "Sobrepeso", "Obesidad I", "Obesidad II", "Obesidad III"]
CORTES_IMC = [-np.inf, 18.5, 25, 30, 35, 40, np.inf]
CL_VARS = ["BMI", "Age", "GenHlth", "MentHlth", "PhysHlth", "Education", "Income", "CardioRisk"]

# Orden exacto de las 17 variables de entrada (todas las columnas del CSV salvo la objetivo)
ENTRADAS = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
            "Age", "Education", "Income"]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta las columnas indicadas a los percentiles aprendidos en fit()."""

    def __init__(self, columnas=("BMI",), inferior=0.01, superior=0.99):
        self.columnas, self.inferior, self.superior = columnas, inferior, superior

    def fit(self, X, y=None):
        self.limites_ = {c: (float(X[c].quantile(self.inferior)), float(X[c].quantile(self.superior)))
                         for c in self.columnas}
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.limites_.items():
            X[c] = X[c].clip(lo, hi)
        return X


def cardio_risk(X: pd.DataFrame) -> pd.Series:
    """Número de condiciones cardiometabólicas (0–4)."""
    return X["HighBP"] + X["HighChol"] + X["HeartDiseaseorAttack"] + X["Stroke"]


def categoria_imc(bmi) -> pd.Series:
    """Categoría de IMC según la OMS."""
    return pd.cut(bmi, bins=CORTES_IMC, right=False, labels=CAT_IMC).astype(str)


def crear_variables(X):
    """Variables derivadas para modelos lineales y MLP (Fase 2)."""
    X = X.copy()
    X["BMI_cat"] = categoria_imc(X["BMI"])
    X["UnhealthyDays"] = np.minimum(30, X["MentHlth"] + X["PhysHlth"])
    X["Age_BMI"] = X["Age"] * X["BMI"]
    return X


def vars_arbol(X):
    """Variables para modelos de árboles: numéricas, sin One-Hot ni escalamiento."""
    X = X.copy().astype(float)
    X["CardioRisk"] = cardio_risk(X)
    X["UnhealthyDays"] = np.minimum(30, X["MentHlth"] + X["PhysHlth"])
    X["Age_BMI"] = X["Age"] * X["BMI"]
    return X


def vars_reg_arbol(X):
    """Variables para la regresión del IMC con árboles (sin BMI_cat ni Age_BMI: evitarían fuga)."""
    X = X.copy().astype(float)
    X["CardioRisk"] = cardio_risk(X)
    X["UnhealthyDays"] = np.minimum(30, X["MentHlth"] + X["PhysHlth"])
    return X


def preprocesador(num, bin_, cat=("BMI_cat",)):
    """Imputación, codificación y escalamiento (se ajusta solo con entrenamiento)."""
    pasos = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), list(num)),
             ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                               ("cod", OrdinalEncoder(categories=ORD_CATS, handle_unknown="use_encoded_value", unknown_value=-1)),
                               ("esc", StandardScaler())]), ORD),
             ("bin", SimpleImputer(strategy="most_frequent"), list(bin_))]
    if cat:
        pasos.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), list(cat)))
    return ColumnTransformer(pasos, remainder="drop", verbose_feature_names_out=False)


def pipe_arbol_clf(modelo):
    """recorte del IMC -> variables derivadas -> modelo de árboles."""
    return Pipeline([("recorte", Winsorizer()), ("derivadas", FunctionTransformer(vars_arbol)), ("modelo", modelo)])


def pipe_reg_arbol(modelo):
    return Pipeline([("derivadas", FunctionTransformer(vars_reg_arbol)), ("modelo", modelo)])


def vars_cluster(X, limites_imc):
    """Ocho variables de segmentación; el IMC se recorta con los límites aprendidos en entrenamiento."""
    X = X.copy().astype(float)
    X["BMI"] = X["BMI"].clip(*limites_imc)
    X["CardioRisk"] = cardio_risk(X)
    return X[CL_VARS]
