# Despliegue del dashboard en Streamlit Community Cloud

Streamlit Community Cloud publica gratis una aplicación Streamlit directamente desde un repositorio público de GitHub. El proceso completo toma unos 10 minutos.

## 1. Publicar el repositorio en GitHub

1. Inicie sesión en https://github.com y cree un repositorio **público** llamado `proyecto-diabetes-brfss2023` (sin README, sin .gitignore: ya vienen incluidos).
2. Descomprima el proyecto y, desde su carpeta, ejecute:

```bash
git init
git add .
git commit -m "Proyecto integrador: diabetes BRFSS 2023 (Fases 1 a 4)"
git remote add origin https://github.com/alainkapra/proyecto-diabetes-brfss2023.git
git branch -M main
git push -u origin main
```

Si GitHub pide contraseña, use un *Personal Access Token* (Settings → Developer settings → Personal access tokens). Alternativa sin consola: GitHub Desktop → File → Add local repository → Publish repository (desmarque «Keep this code private»).

> El archivo más pesado es `data/raw/cdc_brfss_diabetes_2023.csv` (≈ 20 MB), por debajo del límite de 100 MB de GitHub.

## 2. Crear la aplicación

1. Entre a https://share.streamlit.io e inicie sesión con su cuenta de GitHub.
2. Pulse **Create app** → **Deploy a public app from GitHub**.
3. Complete:
   - **Repository:** `alainkapra/proyecto-diabetes-brfss2023`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
   - **App URL:** por ejemplo `diabetes-brfss2023`
4. En **Advanced settings**, elija **Python 3.13**.
5. Pulse **Deploy**. La primera instalación tarda de 3 a 5 minutos.

Streamlit Cloud instala `dashboard/requirements.txt` (está junto al archivo principal), que contiene solo las librerías del dashboard. Así se evita instalar PyTorch, que no se usa en línea.

## 3. Verificar

- Abra la URL asignada y recorra las seis páginas.
- En **Predicción en vivo**, pulse «Alto riesgo»: el medidor debe marcar 91.
- En **Desempeño del modelo**, la tarjeta AUC-ROC debe mostrar 0,816.

## 4. Actualizar los enlaces

Copie la URL definitiva en:
- `README.md` (línea «Dashboard en vivo»).
- El informe final (`reports/FASE_4_Informe_Final_...docx`, Anexo A).

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| Las letras se ven con otra tipografía | Verifique que la carpeta `dashboard/static/fonts` se subió y que `.streamlit/config.toml` tiene `enableStaticServing = true`. |
| `ModuleNotFoundError: src` | Verifique que *Main file path* sea `dashboard/app.py` y que la carpeta `src/` esté en el repositorio. |
| Error al cargar un `.joblib` | Python o scikit-learn/XGBoost distintos a los fijados: elija Python 3.13 y no cambie las versiones de `dashboard/requirements.txt`. Como alternativa, ejecute `python -m src.entrenar_modelos` con sus versiones y suba de nuevo `models/`. |
| La app «se duerme» | Es normal en el plan gratuito tras días sin visitas: pulse *Wake up* y espere unos segundos. Ábrala unos minutos antes de la sustentación. |
