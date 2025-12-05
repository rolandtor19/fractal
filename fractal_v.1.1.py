import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from datetime import datetime, timedelta
import re

# --- CONFIGURACIÓN INICIAL DE LA PÁGINA ---
st.set_page_config(
    page_title="Fractal Hunter Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔮 Radar de Fractales & Backtester (Cross-Market)")
st.markdown("""
**Herramienta de Ingeniería Financiera:** Busca patrones matemáticos idénticos en el pasado de diferentes mercados 
para proyectar movimientos futuros. Incluye **Máquina del Tiempo** para validar estrategias.
""")

# --- LISTAS PREDEFINIDAS ---
COMMON_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", # Cripto
    "QQQ", "SPY", "DIA", "IWM",                 # Índices
    "NVDA", "TSLA", "AAPL", "MSFT", "MSTR",     # Acciones Volátiles
    "GLD", "SLV", "USO", "TLT",                 # Commodities/Bonos
    "EURUSD=X", "JPY=X"                         # Forex
]

COMMON_TFS = ["15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]

# --- BARRA LATERAL: CONTROLES ---
with st.sidebar:
    st.header("1. Configuración del Objetivo")
    
    # Selector de Ticker con opción manual
    mode_ticker = st.radio("Selección de Activo:", ["Lista Común", "Manual"], horizontal=True)
    if mode_ticker == "Lista Común":
        ticker_obj = st.selectbox("Activo Objetivo", COMMON_TICKERS, index=0)
    else:
        ticker_obj = st.text_input("Escribe el Ticker (Yahoo)", value="BTC-USD")
        
    c1, c2 = st.columns(2)
    with c1:
        tf_obj = st.selectbox("Timeframe", COMMON_TFS, index=4) # Default 1d
    with c2:
        columna_analisis = st.selectbox("Precio:", ["Close", "Low", "High"], index=1) # Default Low (Soportes)

    st.divider()
    
    st.header("2. Máquina del Tiempo (Backtest)")
    enable_backtest = st.checkbox("Activar Backtesting", value=False)
    
    fecha_corte = None
    if enable_backtest:
        fecha_corte = st.date_input(
            "Analizar como si hoy fuera:",
            value=datetime.today() - timedelta(days=60),
            max_value=datetime.today()
        )
        st.caption("ℹ️ El algoritmo 'olvidará' todo lo que pasó después de esta fecha.")
    
    st.divider()
    
    st.header("3. Librerías de Búsqueda")
    
    # Librería 1
    l1_c1, l1_c2 = st.columns([2, 1])
    with l1_c1:
        lib1_ticker = st.selectbox("Librería 1", COMMON_TICKERS, index=4) # QQQ default
    with l1_c2:
        lib1_tf = st.selectbox("TF L1", ["1d", "1wk"], index=0)
        
    # Librería 2
    l2_c1, l2_c2 = st.columns([2, 1])
    with l2_c1:
        lib2_ticker = st.selectbox("Librería 2", COMMON_TICKERS, index=13) # GLD default
    with l2_c2:
        lib2_tf = st.selectbox("TF L2", ["1d", "1wk"], index=0)

    st.divider()
    
    st.header("4. Parámetros Matemáticos")
    ventana = st.slider("Memoria (Ventana)", 30, 365, 120)
    proyeccion = st.slider("Proyección (Futuro)", 5, 90, 30)
    resultados = st.slider("Top Coincidencias", 1, 10, 3)
    
    run_btn = st.button("🚀 EJECUTAR ANÁLISIS", type="primary", use_container_width=True)

# --- FUNCIONES DE LÓGICA (CACHÉ) ---

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_y_procesar(ticker, tf, col_target, fecha_limite_str=None):
    """Descarga datos y gestiona el recorte de tiempo para backtesting."""
    
    # 1. Definir periodo de descarga según timeframe
    tf_lower = tf.lower()
    es_intradia = "m" in tf_lower or "h" in tf_lower
    periodo_yahoo = "730d" if es_intradia else "max"
    
    # 2. Descarga
    try:
        data = yf.download(ticker, period=periodo_yahoo, interval=tf, progress=False, auto_adjust=False)
        
        # Selección de columna
        if isinstance(data.columns, pd.MultiIndex):
            data = data[col_target]
        else:
            data = data[col_target]
        
        data = data.dropna()
        
        if len(data) < 50:
            return None, None, None, None, f"Pocos datos para {ticker} ({len(data)} velas)."

        # Limpieza de Zona Horaria
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        # 3. Lógica de Recorte (Backtest)
        vals_futuro = []
        fechas_futuro = []
        
        if fecha_limite_str:
            fecha_dt = pd.to_datetime(fecha_limite_str)
            
            # Cortar Datos
            mask_pasado = data.index <= fecha_dt
            data_pasado = data[mask_pasado]
            
            # Guardar "El Futuro Real" para validación
            mask_futuro = data.index > fecha_dt
            data_futuro = data[mask_futuro]
            
            vals_pasado = data_pasado.values.flatten()
            fechas_pasado = data_pasado.index
            
            if not data_futuro.empty:
                vals_futuro = data_futuro.values.flatten()
                fechas_futuro = data_futuro.index
                
            return vals_pasado, fechas_pasado, vals_futuro, fechas_futuro, None
            
        else:
            # Modo Live (Sin recorte)
            return data.values.flatten(), data.index, [], [], None

    except Exception as e:
        return None, None, None, None, str(e)

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

    # Bucle de búsqueda (Vectorizar esto sería el siguiente nivel de optimización)
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

# --- LÓGICA DE EJECUCIÓN PRINCIPAL ---

if run_btn:
    # Contenedor de estado
    status_text = st.empty()
    bar = st.progress(0)
    
    try:
        # A. DESCARGA DE DATOS
        status_text.text("📡 Descargando datos de mercado...")
        
        # Fecha límite (si aplica)
        fecha_str = fecha_corte.strftime('%Y-%m-%d') if enable_backtest else None
        
        # 1. Objetivo
        obj_p, obj_f, real_p, real_f, err = descargar_y_procesar(ticker_obj, tf_obj, columna_analisis, fecha_str)
        if err:
            st.error(f"Error Objetivo: {err}")
            st.stop()
            
        if len(obj_p) < ventana:
            st.error(f"Historial insuficiente en {ticker_obj} antes de la fecha de corte. Necesitas {ventana} velas.")
            st.stop()
            
        patron_actual = obj_p[-ventana:]
        bar.progress(20)
        
        # 2. Librerías (Usamos fecha de corte también para no tener "look-ahead bias" estricto, 
        # aunque a veces es útil buscar en el futuro de otro activo si es cross-market. 
        # Por rigor, cortaremos también las librerías a la fecha de corte).
        
        lib1_p, lib1_f, _, _, err1 = descargar_y_procesar(lib1_ticker, lib1_tf, columna_analisis, fecha_str)
        lib2_p, lib2_f, _, _, err2 = descargar_y_procesar(lib2_ticker, lib2_tf, columna_analisis, fecha_str)
        
        bar.progress(40)
        
        # B. CÁLCULO MATEMÁTICO
        status_text.text("🧮 Calculando distancias euclidianas y fractales...")
        
        matches = []
        if lib1_p is not None: matches += escanear_libreria(lib1_ticker, lib1_p, lib1_f, patron_actual, proyeccion)
        if lib2_p is not None: matches += escanear_libreria(lib2_ticker, lib2_p, lib2_f, patron_actual, proyeccion)
        
        bar.progress(70)
        
        if not matches:
            st.warning("No se encontraron patrones similares. Intenta reducir la ventana o cambiar de activo.")
            st.stop()
            
        # Ordenar y Filtrar
        matches.sort(key=lambda x: x['score'])
        
        seleccionados = []
        indices_usados = {} 
        distancia_min = int(ventana * 0.6)
        
        for m in matches:
            src = m['source']
            idx = m['indice']
            if src not in indices_usados: indices_usados[src] = []
            
            repetido = False
            for usado in indices_usados[src]:
                if abs(idx - usado) < distancia_min:
                    repetido = True
                    break
            
            if not repetido:
                seleccionados.append(m)
                indices_usados[src].append(idx)
            
            if len(seleccionados) >= resultados: break
            
        bar.progress(90)
        
        # C. CONSTRUCCIÓN DE PROYECCIÓN
        status_text.text("🎨 Generando visualización...")
        
        suma_proyecciones = np.zeros(proyeccion)
        suma_pesos = 0
        patron_actual_norm = normalizar(patron_actual)
        ultimo_valor_actual = patron_actual_norm[-1]
        
        series_graficar = []
        
        for match in seleccionados:
            serie_comp = np.concatenate([match['datos_past'], match['datos_fut']])
            serie_norm = normalizar(serie_comp)
            
            # Alineación Pivot
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
        bar.progress(100)
        status_text.empty()
        
        # --- D. PLOT CON MATPLOTLIB ---
        # Usamos el estilo oscuro de Streamlit por defecto, así que ajustamos colores para contraste
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        x_pasado = np.arange(-ventana + 1, 1)
        x_futuro = np.arange(1, proyeccion + 1)
        x_total = np.concatenate([x_pasado, x_futuro])
        
        # 1. Fantasmas
        for s in series_graficar:
            color = 'steelblue' if s['source'] == lib1_ticker else 'chocolate'
            ax.plot(x_total, s['serie'], label=s['label'], color=color, alpha=0.4, linewidth=1.2)
            
        # 2. Proyección Maestra
        y_master = np.insert(linea_maestra, 0, ultimo_valor_actual)
        x_master = np.insert(x_futuro, 0, 0)
        ax.plot(x_master, y_master, label="PROYECCIÓN HÍBRIDA", color='#00ff00', linewidth=3.5, zorder=10)
        
        # 3. Actual (Historia Reciente)
        ax.plot(x_pasado, patron_actual_norm, label=f"ACTUAL ({ticker_obj})", color='black', linewidth=2.5, zorder=11)
        
        # 4. REALIDAD (Solo si es Backtest)
        if enable_backtest and len(real_p) > 0:
            # Normalizar realidad con la escala del patrón original
            min_p = np.min(patron_actual)
            max_p = np.max(patron_actual)
            rng = max_p - min_p
            
            realidad_norm = (real_p - min_p) / rng
            
            # Recortar si la realidad es más larga que la proyección
            limit_len = min(len(realidad_norm), proyeccion)
            y_real = realidad_norm[:limit_len]
            x_real = np.arange(1, limit_len + 1)
            
            # Insertar punto 0 para conectar
            y_real_con = np.insert(y_real, 0, ultimo_valor_actual)
            x_real_con = np.insert(x_real, 0, 0)
            
            ax.plot(x_real_con, y_real_con, label="REALIDAD (VALIDACIÓN)", color='white', linewidth=2.5, linestyle='--', zorder=12)

        # === CONFIGURACIÓN DE EJES ===
        ax.set_xlim(x_total[0], x_total[-1])
        locator = ticker.MaxNLocator(nbins=25, integer=True)
        ax.xaxis.set_major_locator(locator)
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.3)
        ax.set_xlabel(f"Velas de {tf_obj} (Pasado <--- 0 ---> Futuro)", fontsize=10, color='gray')
        
        # Eje Superior (Tiempo)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.xaxis.set_major_locator(locator)
        
        def obtener_delta(tf_str):
            num_match = re.search(r'\d+', tf_str)
            num = int(num_match.group()) if num_match else 1
            unit = tf_str.lower()
            if 'wk' in unit: return pd.Timedelta(weeks=num)
            if 'mo' in unit: return pd.Timedelta(days=30*num)
            if 'm' in unit and 'o' not in unit: return pd.Timedelta(minutes=num)
            if 'h' in unit: return pd.Timedelta(hours=num)
            return pd.Timedelta(days=1)

        delta = obtener_delta(tf_obj)
        ref_date = obj_f[-1] # Última fecha cargada (sea hoy o fecha de corte)
        es_intradia_plot = "m" in tf_obj or "h" in tf_obj

        def date_fmt(x, pos):
            dt = ref_date + (delta * x)
            if es_intradia_plot: return dt.strftime("%d-%b %Hh")
            else: return dt.strftime("%d-%b")

        ax_top.xaxis.set_major_formatter(ticker.FuncFormatter(date_fmt))
        ax_top.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Eje Precio (Derecho)
        ax2 = ax.twinx()
        min_p, max_p = np.min(patron_actual), np.max(patron_actual)
        rng = max_p - min_p
        y1, y2 = ax.get_ylim()
        ax2.set_ylim(y1 * rng + min_p, y2 * rng + min_p)
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
        ax2.set_ylabel(f"Precio {ticker_obj}", fontweight='bold')
        
        curr_price = patron_actual[-1]
        ax2.axhline(curr_price, color='#444444', ls='--', lw=1.5, alpha=0.8)
        ax2.text(x_pasado[0], curr_price, f" ${curr_price:,.2f} ", 
                 color='black', fontweight='bold', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Títulos
        label_fecha = "Fecha Corte" if enable_backtest else "Ahora"
        plt.axvline(0, color='red', ls=':', label=label_fecha)
        
        titulo = f"Fractalidad: {ticker_obj} ({tf_obj}) vs [{lib1_ticker} & {lib2_ticker}]"
        if enable_backtest:
            titulo += f" | BACKTEST: {fecha_str}"
            
        plt.title(titulo, pad=25, fontsize=14)
        ax.legend(bbox_to_anchor=(1.08, 1), loc='upper left', fontsize=8)
        
        # Renderizar en Streamlit
        st.pyplot(fig)
        
        # --- TABLA DE DATOS (Opcional) ---
        with st.expander("📊 Ver Datos de Proyección"):
            # Generar fechas futuras
            fechas_proy = [ref_date + (delta * i) for i in range(1, proyeccion + 1)]
            # Des-normalizar precios
            precios_proy = (linea_maestra * rng) + min_p
            
            df_proy = pd.DataFrame({
                "Fecha": fechas_proy,
                "Precio Estimado": precios_proy
            })
            st.dataframe(df_proy)

    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")