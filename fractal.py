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
            ax.plot(x_total, s['serie'], label=s['label'], color=color, alpha=0.4, linewidth=1.2)
            
        # Maestra
        y_master = np.insert(linea_maestra, 0, ultimo_valor_actual)
        x_master = np.insert(x_futuro, 0, 0)
        ax.plot(x_master, y_master, label="PROYECCIÓN HÍBRIDA", color='#00ff00', linewidth=3.5, zorder=10)
        
        # Actual
        ax.plot(x_pasado, patron_actual_norm, label=f"ACTUAL ({ticker_obj})", color='black', linewidth=2.5, zorder=11)
        
        # Ejes
        ax.set_xlim(x_total[0], x_total[-1])
        locator = ticker.MaxNLocator(nbins=25, integer=True)
        ax.xaxis.set_major_locator(locator)
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.3)
        ax.set_xlabel(f"Velas de {tf_obj} (Pasado <--- 0 ---> Futuro)", fontsize=10, color='gray')
        
        # Eje Superior Tiempo
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.xaxis.set_major_locator(locator)
        
        def obtener_delta(tf_str):
            num = int(re.search(r'\d+', tf_str).group()) if re.search(r'\d+', tf_str) else 1
            unit = tf_str.lower()
            if 'wk' in unit: return pd.Timedelta(weeks=num)
            if 'mo' in unit: return pd.Timedelta(days=30*num)
            if 'm' in unit and 'o' not in unit: return pd.Timedelta(minutes=num)
            if 'h' in unit: return pd.Timedelta(hours=num)
            return pd.Timedelta(days=1)

        delta = obtener_delta(tf_obj)
        ref_date = fechas_obj[-1]
        es_intradia_plot = "m" in tf_obj or "h" in tf_obj

        def date_fmt(x, pos):
            dt = ref_date + (delta * x)
            if es_intradia_plot: return dt.strftime("%d-%b %Hh")
            else: return dt.strftime("%d-%b")

        ax_top.xaxis.set_major_formatter(ticker.FuncFormatter(date_fmt))
        ax_top.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Eje Precio
        ax2 = ax.twinx()
        min_p, max_p = np.min(patron_actual), np.max(patron_actual)
        rng = max_p - min_p
        y1, y2 = ax.get_ylim()
        ax2.set_ylim(y1 * rng + min_p, y2 * rng + min_p)
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
        ax2.set_ylabel(f"Precio {ticker_obj}", fontweight='bold')
        
        curr_price = patron_actual[-1]
        ax2.axhline(curr_price, color='#444444', ls='--', lw=1.5, alpha=0.8)
        ax2.text(x_pasado[0], curr_price, f" Precio Actual: ${curr_price:,.2f} ", 
                 color='black', fontweight='bold', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.axvline(0, color='red', ls=':', label="Ahora")
        plt.title(f"Fractalidad Cruzada: {ticker_obj} ({tf_obj}) vs [{lib1_ticker} & {lib2_ticker}]", pad=20, fontsize=14)
        ax.legend(bbox_to_anchor=(1.08, 1), loc='upper left', fontsize=8)
        
        st.pyplot(fig)
        
        st.success("Análisis finalizado exitosamente.")
