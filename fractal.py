import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import re

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Fractal Hunter Pro", page_icon="📈")

st.title("🛰️ Radar de Fractales Intermercado (Cross-Market)")
st.markdown("""
Esta herramienta busca patrones matemáticos similares en el pasado de diferentes activos 
para proyectar movimientos futuros basados en la psicología de masas.
""")

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("1. Patrón Objetivo (Actual)")
    ticker_obj = st.text_input("Ticker Objetivo", value="BTC-USD")
    tf_obj = st.selectbox("Timeframe Objetivo", 
                          ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], 
                          index=6) # Default 1d
    columna_analisis = st.selectbox("Analizar Precio:", ["Close", "Low", "High"], index=0)
    
    st.divider()
    
    st.header("2. Librerías Históricas")
    st.info("¿Dónde buscamos patrones del pasado?")
    lib1_ticker = st.text_input("Librería 1 (Principal)", value="QQQ")
    lib1_tf = st.selectbox("Timeframe Lib 1", ["1d", "1wk"], index=0)
    
    lib2_ticker = st.text_input("Librería 2 (Secundaria)", value="GLD")
    lib2_tf = st.selectbox("Timeframe Lib 2", ["1d", "1wk"], index=0)
    
    st.divider()
    
    st.header("3. Parámetros del Fractal")
    ventana = st.slider("Memoria (Velas Atrás)", 30, 365, 120)
    proyeccion = st.slider("Proyección (Velas Futuro)", 5, 60, 30)
    resultados = st.slider("Top Coincidencias", 1, 10, 3)
    
    run_btn = st.button("🔍 ESCANEAR MERCADO", type="primary")

# --- LÓGICA DE DATOS (CACHÉ) ---
@st.cache_data(ttl=3600) # Guarda en memoria por 1 hora
def descargar_data(ticker, tf, columna):
    # Reglas de límite de Yahoo
    tf_lower = tf.lower()
    if tf_lower == "1m": periodo = "7d"
    elif any(x in tf_lower for x in ["2m","5m","15m","30m","90m"]): periodo = "59d"
    elif any(x in tf_lower for x in ["60m","1h"]): periodo = "730d"
    else: periodo = "max"
    
    try:
        data = yf.download(ticker, period=periodo, interval=tf, progress=False, auto_adjust=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data = data[columna]
        else:
            data = data[columna]
        
        data = data.dropna()
        
        # Validación mínima
        if len(data) < 50:
            return None, None, f"Pocos datos para {ticker} ({len(data)} velas)"
            
        # Limpieza de zona horaria
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        return data.values.flatten(), data.index, None
        
    except Exception as e:
        return None, None, str(e)

# --- FUNCIONES MATEMÁTICAS ---
def normalizar(array):
    if len(array) == 0: return array
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val == min_val: return array
    return (array - min_val) / (max_val - min_val)

def escanear_libreria(nombre_lib, precios_lib, fechas_lib, patron_target, n_proyeccion):
    len_patron = len(patron_target)
    patron_norm = normalizar(patron_target)
    hallazgos = []
    
    tope = len(precios_lib) - len_patron - n_proyeccion
    if tope <= 0: return []

    # Optimización: Convertir a matriz numpy para velocidad si fuera muy grande
    # Por ahora mantenemos el loop simple por claridad
    for i in range(tope):
        candidato = precios_lib[i : i + len_patron]
        candidato_norm = normalizar(candidato)
        score = np.linalg.norm(patron_norm - candidato_norm)
        
        hallazgos.append({
            'source': nombre_lib,          
            'indice': i,
            'fecha_origen': fechas_lib[i], 
            'score': score,
            'datos_past': candidato,       
            'datos_fut': precios_lib[i + len_patron : i + len_patron + n_proyeccion] 
        })
    return hallazgos

# --- LÓGICA DE EJECUCIÓN ---
if run_btn:
    with st.spinner(f'Descargando y analizando datos de {ticker_obj}, {lib1_ticker} y {lib2_ticker}...'):
        
        # 1. Descarga Objetivo
        precios_obj, fechas_obj, err = descargar_data(ticker_obj, tf_obj, columna_analisis)
        if err:
            st.error(f"Error en Objetivo: {err}")
            st.stop()
            
        if len(precios_obj) < (ventana + 5):
            st.error(f"Historial insuficiente en {ticker_obj}. Se necesitan {ventana} velas.")
            st.stop()
            
        patron_actual = precios_obj[-ventana:]
        
        # 2. Descarga Librerías
        lib1_p, lib1_f, err1 = descargar_data(lib1_ticker, lib1_tf, columna_analisis)
        lib2_p, lib2_f, err2 = descargar_data(lib2_ticker, lib2_tf, columna_analisis)
        
        if err1: st.warning(f"Librería 1 falló: {err1}")
        if err2: st.warning(f"Librería 2 falló: {err2}")
        
        # 3. Escaneo
        matches = []
        if lib1_p is not None:
            matches += escanear_libreria(lib1_ticker, lib1_p, lib1_f, patron_actual, proyeccion)
        if lib2_p is not None:
            matches += escanear_libreria(lib2_ticker, lib2_p, lib2_f, patron_actual, proyeccion)
            
        # 4. Filtrado y Ranking
        if not matches:
            st.error("No se encontraron coincidencias. Intenta reducir la ventana o cambiar de activo.")
            st.stop()
            
        matches.sort(key=lambda x: x['score'])
        
        seleccionados = []
        indices_usados = {} # {source: [indices]}
        distancia_min = int(ventana * 0.6)
        
        for m in matches:
            src = m['source']
            idx = m['indice']
            if src not in indices_usados: indices_usados[src] = []
            
            # Verificar colisión
            repetido = False
            for usado in indices_usados[src]:
                if abs(idx - usado) < distancia_min:
                    repetido = True
                    break
            
            if not repetido:
                seleccionados.append(m)
                indices_usados[src].append(idx)
            
            if len(seleccionados) >= resultados: break
            
        # 5. Cálculo Proyección
        suma_proyecciones = np.zeros(proyeccion)
        suma_pesos = 0
        patron_actual_norm = normalizar(patron_actual)
        ultimo_valor_actual = patron_actual_norm[-1]
        
        series_graficar = []
        
        for match in seleccionados:
            serie_comp = np.concatenate([match['datos_past'], match['datos_fut']])
            serie_norm = normalizar(serie_comp)
            
            punto_empalme = serie_norm[-(proyeccion+1)]
            offset = ultimo_valor_actual - punto_empalme
            serie_alineada = serie_norm + offset
            
            peso = 1 / (match['score']**2 + 0.0001)
            suma_proyecciones += serie_alineada[-proyeccion:] * peso
            suma_pesos += peso
            
            series_graficar.append({
                'serie': serie_alineada,
                'source': match['source'],
                'label': f"{match['source']} | {match['fecha_origen'].strftime('%Y-%m-%d')} (Score: {match['score']:.2f})"
            })
            
        linea_maestra = suma_proyecciones / suma_pesos
        
        # --- VISUALIZACIÓN ---
        # Usamos st.pyplot para renderizar Matplotlib
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x_pasado = np.arange(-ventana + 1, 1)
        x_futuro = np.arange(1, proyeccion + 1)
        x_total = np.concatenate([x_pasado, x_futuro])
        
        # Fantasmas
        for s in series_graficar:
            color = 'steelblue' if s['source'] == lib1_ticker else 'chocolate'
            ax.plot(x_total, s['serie'], label=s['label'], color=color