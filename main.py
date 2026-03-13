"""
PASO 2: API de predicción — se despliega en Render.com
Este archivo NO se ejecuta en tu PC, va en Render.

Endpoints disponibles:
  GET  /                          → health check
  POST /predecir-condicion        → predice próxima condición de UN activo
  GET  /ranking-riesgo            → top activos con mayor probabilidad de falla
  GET  /resumen-activo/{nombre}   → detalle de un activo específico
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import os

app = FastAPI(
    title="API Predicción Condición Activos",
    description="Predice la próxima condición de activos industriales basado en histórico de vibraciones y temperatura",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CARGA DEL MODELO ────────────────────────────────────────────────────────
MODEL_PATH = "modelo_activos.pkl"
paquete = None

@app.on_event("startup")
async def cargar_modelo():
    global paquete
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Archivo {MODEL_PATH} no encontrado. Los endpoints devolverán error.")
        return
    paquete = joblib.load(MODEL_PATH)
    print(f"✅ Modelo cargado. Activos en base: {len(paquete['stats_activos'])}")
    print(f"   Clases: {paquete['clases']}")

# ─── MODELOS DE DATOS ────────────────────────────────────────────────────────
class InputPrediccion(BaseModel):
    activo: str                        # Nombre del activo (ej: "BOMBA ACEITE SELLO -A")
    vrms: float                        # Velocidad vibración mm/s RMS
    temperatura: float                 # Temperatura °C
    condicion_actual: Optional[str] = None   # Si se conoce: BUENO, SATISFACTORIO, etc.
    ruta: Optional[str] = None
    ubicacion: Optional[str] = None
    criticidad: Optional[str] = "A"
    dir_medicion: Optional[str] = "1H"

# ─── FUNCIÓN AUXILIAR ────────────────────────────────────────────────────────
def construir_features(input_data: InputPrediccion):
    """Construye el dataframe de features para el modelo."""
    stats = paquete['stats_activos']
    features_cat = paquete['features_cat']
    features_num = paquete['features_num']

    # Buscar estadísticas históricas del activo
    activo_upper = input_data.activo.upper().strip()
    hist = stats[stats['Activo'].str.upper() == activo_upper]

    if len(hist) > 0:
        h = hist.iloc[0]
        vrms_prom = float(h['vrms_promedio'])
        vrms_max = float(h['vrms_max'])
        temp_prom = float(h['temp_promedio'])
        tasa_falla = float(h['tasa_falla'])
        ruta = input_data.ruta or str(h['ruta'])
        ubicacion = input_data.ubicacion or str(h['ubicacion'])
        criticidad = input_data.criticidad or str(h['criticidad'])
        dir_med = input_data.dir_medicion or str(h['dir_principal'])
        condicion_act = input_data.condicion_actual or str(h['ultima_condicion'])
    else:
        # Activo nuevo: usar valores del input y promedios generales
        vrms_prom = input_data.vrms
        vrms_max = input_data.vrms
        temp_prom = input_data.temperatura
        tasa_falla = 0.1
        ruta = input_data.ruta or "DESCONOCIDO"
        ubicacion = input_data.ubicacion or "DESCONOCIDO"
        criticidad = input_data.criticidad or "A"
        dir_med = input_data.dir_medicion or "1H"
        condicion_act = input_data.condicion_actual or "BUENO"

    row = {
        'Ruta': ruta.upper(),
        'Ubicación': ubicacion.upper(),
        'Criticidad': criticidad.upper(),
        'Dir': dir_med.upper(),
        'Condicion_Simple': condicion_act.upper(),
        'V/rms': input_data.vrms,
        'T°': input_data.temperatura,
        'vrms_promedio': vrms_prom,
        'vrms_max': vrms_max,
        'temp_promedio': temp_prom,
        'tasa_falla_historica': tasa_falla,
        'dias_desde_ultima': 30.0,
    }

    return pd.DataFrame([row])[features_cat + features_num]

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    modelo_ok = paquete is not None
    return {
        "status": "ok",
        "modelo_cargado": modelo_ok,
        "activos_en_base": len(paquete['stats_activos']) if modelo_ok else 0,
        "clases_prediccion": paquete['clases'] if modelo_ok else []
    }


@app.post("/predecir-condicion")
def predecir_condicion(data: InputPrediccion):
    """
    Predice la próxima condición de un activo dado su V/rms y temperatura actual.
    Retorna la condición predicha y las probabilidades de cada clase.
    """
    if paquete is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        X = construir_features(data)

        # Predicción de condición
        condicion_pred = paquete['modelo_condicion'].predict(X)[0]
        probs_condicion = paquete['modelo_condicion'].predict_proba(X)[0]
        clases = paquete['clases']

        # Probabilidad de falla
        prob_falla = paquete['modelo_riesgo'].predict_proba(X)[0][1]

        # Nivel de alerta
        if prob_falla >= 0.7:
            alerta = "CRÍTICA"
        elif prob_falla >= 0.4:
            alerta = "ADVERTENCIA"
        else:
            alerta = "NORMAL"

        return {
            "activo": data.activo,
            "condicion_actual": data.condicion_actual or "No informada",
            "vrms_actual": data.vrms,
            "temperatura_actual": data.temperatura,
            "proxima_condicion_predicha": condicion_pred,
            "probabilidad_falla": round(float(prob_falla), 4),
            "porcentaje_riesgo": f"{prob_falla * 100:.1f}%",
            "nivel_alerta": alerta,
            "probabilidades_condicion": {
                clase: round(float(prob), 4)
                for clase, prob in zip(clases, probs_condicion)
            },
            "recomendacion": _generar_recomendacion(condicion_pred, prob_falla)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ranking-riesgo")
def ranking_riesgo(top: int = 20, ubicacion: Optional[str] = None):
    """
    Retorna los N activos con mayor probabilidad de pasar a condición INACEPTABLE.
    Ideal para mostrar en Power BI como tabla de riesgo.
    """
    if paquete is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        stats = paquete['stats_activos'].copy()

        # Filtrar por ubicación si se especifica
        if ubicacion:
            stats = stats[stats['ubicacion'].str.upper() == ubicacion.upper()]

        if len(stats) == 0:
            return {"activos": [], "total": 0}

        features_cat = paquete['features_cat']
        features_num = paquete['features_num']

        resultados = []
        for _, row in stats.iterrows():
            try:
                feat_row = {
                    'Ruta': str(row['ruta']).upper(),
                    'Ubicación': str(row['ubicacion']).upper(),
                    'Criticidad': str(row['criticidad']).upper(),
                    'Dir': str(row['dir_principal']).upper(),
                    'Condicion_Simple': str(row['ultima_condicion']).upper(),
                    'V/rms': float(row['vrms_promedio']),
                    'T°': float(row['temp_promedio']),
                    'vrms_promedio': float(row['vrms_promedio']),
                    'vrms_max': float(row['vrms_max']),
                    'temp_promedio': float(row['temp_promedio']),
                    'tasa_falla_historica': float(row['tasa_falla']),
                    'dias_desde_ultima': 30.0,
                }
                X = pd.DataFrame([feat_row])[features_cat + features_num]
                prob_falla = float(paquete['modelo_riesgo'].predict_proba(X)[0][1])
                condicion_pred = paquete['modelo_condicion'].predict(X)[0]

                if prob_falla >= 0.7:
                    alerta = "CRÍTICA"
                elif prob_falla >= 0.4:
                    alerta = "ADVERTENCIA"
                else:
                    alerta = "NORMAL"

                resultados.append({
                    "activo": row['Activo'],
                    "ubicacion": str(row['ubicacion']),
                    "criticidad": str(row['criticidad']),
                    "ultima_condicion": str(row['ultima_condicion']),
                    "condicion_predicha": condicion_pred,
                    "probabilidad_falla": round(prob_falla, 4),
                    "porcentaje_riesgo": f"{prob_falla * 100:.1f}%",
                    "nivel_alerta": alerta,
                    "vrms_promedio": round(float(row['vrms_promedio']), 3),
                    "temp_promedio": round(float(row['temp_promedio']), 1),
                    "n_mediciones": int(row['n_mediciones']),
                })
            except:
                continue

        resultados.sort(key=lambda x: x['probabilidad_falla'], reverse=True)

        return {
            "activos": resultados[:top],
            "total_analizados": len(resultados),
            "criticos": sum(1 for r in resultados if r['nivel_alerta'] == 'CRÍTICA'),
            "advertencia": sum(1 for r in resultados if r['nivel_alerta'] == 'ADVERTENCIA'),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resumen-activo/{nombre_activo}")
def resumen_activo(nombre_activo: str):
    """
    Retorna el perfil histórico y predicción de un activo específico.
    """
    if paquete is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    stats = paquete['stats_activos']
    nombre_upper = nombre_activo.upper().strip()
    hist = stats[stats['Activo'].str.upper() == nombre_upper]

    if len(hist) == 0:
        raise HTTPException(status_code=404, detail=f"Activo '{nombre_activo}' no encontrado en base histórica")

    h = hist.iloc[0]
    return {
        "activo": nombre_activo,
        "ubicacion": str(h['ubicacion']),
        "criticidad": str(h['criticidad']),
        "n_mediciones_historicas": int(h['n_mediciones']),
        "vrms_promedio_historico": round(float(h['vrms_promedio']), 3),
        "vrms_maximo_historico": round(float(h['vrms_max']), 3),
        "temp_promedio_historica": round(float(h['temp_promedio']), 1),
        "tasa_falla_historica": f"{float(h['tasa_falla']) * 100:.1f}%",
        "ultima_condicion_registrada": str(h['ultima_condicion']),
    }


@app.get("/activos")
def listar_activos(ubicacion: Optional[str] = None):
    """Lista todos los activos disponibles en la base histórica."""
    if paquete is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    stats = paquete['stats_activos']
    if ubicacion:
        stats = stats[stats['ubicacion'].str.upper() == ubicacion.upper()]

    return {
        "activos": sorted(stats['Activo'].tolist()),
        "total": len(stats)
    }


def _generar_recomendacion(condicion_pred: str, prob_falla: float) -> str:
    if condicion_pred == "INACEPTABLE" or prob_falla >= 0.7:
        return "Intervención urgente recomendada. Programar mantenimiento correctivo."
    elif condicion_pred == "INSATISFACTORIO" or prob_falla >= 0.4:
        return "Monitoreo intensivo recomendado. Evaluar mantenimiento preventivo."
    elif condicion_pred == "SATISFACTORIO":
        return "Continuar monitoreo según ruta. Sin acción inmediata."
    else:
        return "Activo en buen estado. Mantener frecuencia de monitoreo normal."
