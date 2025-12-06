import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from datetime import datetime, timedelta
import re

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Fractal Hunter Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔮 Radar de Fractales & Backtester (Cross-Market)")
st.markdown("""
**Herramienta de Ingeniería Financiera:** Busca patrones matemáticos idénticos en el pasado para proyectar el futuro.
""")

# --- LISTAS PREDEFINIDAS ---
COMMON_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD",
    "QQQ", "SPY", "DIA", "IWM", "^VIX",
    "NVDA", "TSLA", "AAPL", "MSFT", "MSTR", "COIN", "AMD",
    "GLD", "SLV", "USO", "TLT",
    "EURUSD=X", "JPY=X", "GBPUSD=X"
]

COMMON_TFS = ["15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]

# --- HELPER PARA SELECCIÓN DE ACTIVOS ---
def render_asset_selector(label, key_prefix, default_val):
    """Crea un selector híbrido (Lista o Manual)"""
    col_mode, col_input = st.columns([1, 2])
    with col_mode:
        mode = st.radio("Modo", ["Lista", "Manual"], horizontal=True, key=f"{key_prefix}_mode", label_visibility="collapsed")
    
    with col_input:
        if mode == "Lista":
            # Intentamos encontrar el default en la lista, si no, usamos el primero
            idx = COMMON_TICKERS.index(default_val) if default_val in COMMON_TICKERS else 0
            val = st.selectbox(label, COMMON_TICKERS, index=idx, key=f"{key_prefix}_list")
        else:
            val = st.text_input(label, value=default_val, key=f"{key_prefix}_text")
    return val

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1. Configuración del Objetivo")
    # Selector Híbrido para Objetivo
    ticker_obj = render_asset_selector("Activo Objetivo", "target", "BTC-USD")
        
    c1, c2 = st.columns(2)
    with c1:
        tf_obj = st.selectbox("Timeframe", COMMON_TFS, index=4) # Default 1d
    with c2:
        columna_analisis = st.selectbox("Precio:", ["Close", "Low", "High"], index=1) 

    st.divider()
    
    st.header("2. Máquina del Tiempo")
    enable_backtest = st.checkbox("Activar Backtesting", value=False)
    
    fecha_corte = datetime.today()
    if enable_backtest:
        fecha_corte = st.date_input(
            "Fecha de Corte:",
            value=datetime.today() - timedelta(days=60),
            max_value=datetime.today()
        )
        st.info(f"Analizando como si fuera: {fecha_corte.strftime('%Y-%m-%d')}")
    
    st.divider()
    
    st.header("3. Librerías de Búsqueda")
    
    # Librería 1
    st.caption("Librería 1 (Principal)")
    lib1_ticker = render_asset_selector("Lib 1", "lib1", "QQQ")
    lib1_tf = st.selectbox("TF Lib 1", ["1d", "1wk"], index=0)
    
    st.write("") # Espacio
    
    # Librería 2
    st.caption("Librería 2 (Secundaria)")
    lib2_ticker = render_asset_selector("Lib 2", "lib2", "GLD")
    lib2_tf = st.selectbox("TF Lib 2", ["1d", "1wk"], index=0)

    st.divider()
    
    st.header("4. Parámetros")
    ventana = st.slider("Memoria (Velas)", 30, 365, 120)
    proyeccion = st.slider("Proyección (Futuro)", 5, 90, 30)
    resultados = st.slider("Top Coincidencias", 1, 10, 5) # Aumentado default a 5 para ver colores
    
    run_btn = st.button("🚀 EJECUTAR ANÁLISIS", type="primary", use_container_width=True)

# --- FUNCIONES DE LÓGICA ---

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_y_procesar(ticker, tf, col_target, fecha_limite_str=None):
    tf_lower = tf.lower()
    if tf_lower == "1m": periodo = "7d"
    elif any(x in tf_lower for x in ["2m","5m","15m","30m","90m"]): periodo = "59d"
    elif any(x in tf_lower for x in ["60m","1h","4h"]): periodo = "730d"
    else: periodo = "max"
    
    try:
        data = yf.download(ticker, period=periodo, interval=tf, progress=False, auto_adjust=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data = data[col_target]
        else:
            data = data[col_target]
        
        data = data.dropna()
        if len(data) < 50: return None, None, None, None, f"Pocos datos para {ticker}"

        if data.index.tz is not None: data.index = data.index.tz_localize(None)
            
        vals_futuro, fechas_futuro = [], []
        
        if fecha_limite_str:
            fecha_dt = pd.to_datetime(fecha_limite_str)
            mask_pasado = data.index <= fecha_dt
            data_pasado = data[mask_pasado]
            mask_futuro = data.index > fecha_dt
            data_futuro = data[mask_futuro]
            
            vals_pasado = data_pasado.values.flatten()
            fechas_pasado = data_pasado.index
            
            if not data_futuro.empty:
                vals_futuro = data_futuro.values.flatten()
                fechas_futuro = data_futuro.index
                
            return vals_pasado, fechas_pasado, vals_futuro, fechas_futuro, None
        else:
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

# --- EJECUCIÓN ---

if run_btn:
    status_text = st.empty()
    bar = st.progress(0)
    
    try:
        status_text.text("📡 Descargando datos...")
        fecha_str = fecha_corte.strftime('%Y-%m-%d') if enable_backtest else None
        
        # 1. Objetivo
        obj_p, obj_f, real_p, real_f, err = descargar_y_procesar(ticker_obj, tf_obj, columna_analisis, fecha_str)
        if err: st.error(err); st.stop()
        if len(obj_p) < ventana: st.error("Historial insuficiente."); st.stop()
        patron_actual = obj_p[-ventana:]
        bar.progress(20)
        
        # 2. Librerías
        lib1_p, lib1_f, _, _, _ = descargar_y_procesar(lib1_ticker, lib1_tf, columna_analisis, fecha_str)
        lib2_p, lib2_f, _, _, _ = descargar_y_procesar(lib2_ticker, lib2_tf, columna_analisis, fecha_str)
        
        # 3. Escaneo
        status_text.text("🧮 Analizando fractales...")
        matches = []
        if lib1_p is not None: matches += escanear_libreria(lib1_ticker, lib1_p, lib1_f, patron_actual, proyeccion)
        if lib2_p is not None: matches += escanear_libreria(lib2_ticker, lib2_p, lib2_f, patron_actual, proyeccion)
        
        if not matches: st.warning("No se encontraron patrones."); st.stop()
            
        matches.sort(key=lambda x: x['score'])
        
        # 4. Filtrado
        seleccionados = []
        indices_usados = {} 
        distancia_min = int(ventana * 0.6)
        
        for m in matches:
            src = m['source']
            idx = m['indice']
            if src not in indices_usados: indices_usados[src] = []
            
            repetido = False
            for usado in indices_usados[src]:
                if abs(idx - usado) < distancia_min: repetido = True; break
            
            if not repetido:
                seleccionados.append(m)
                indices_usados[src].append(idx)
            
            if len(seleccionados) >= resultados: break
            
        bar.progress(80)
        
        # 5. Construcción Visual
        status_text.text("🎨 Generando gráfico...")
        
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
                'fecha': match['fecha_origen'].strftime('%Y-%m-%d'),
                'score': match['score'],
                'label': f"{match['source']} ({match['fecha_origen'].strftime('%Y-%m-%d')}) Score: {match['score']:.2f}"
            })
            
        linea_maestra = suma_proyecciones / suma_pesos
        bar.progress(100)
        status_text.empty()
        
        # --- PLOTEO ---
        fig, ax = plt.subplots(figsize=(16, 9))
        
        x_pasado = np.arange(-ventana + 1, 1)
        x_futuro = np.arange(1, proyeccion + 1)
        x_total = np.concatenate([x_pasado, x_futuro])
        
        # ----------------------------------------------------
        # MEJORA VISUAL: COLORES ÚNICOS Y ETIQUETAS DE SCORE
        # ----------------------------------------------------
        
        # Generar paleta de colores distintos (tab10 tiene 10 colores distintos)
        colores = plt.cm.tab10(np.linspace(0, 1, len(series_graficar)))
        
        for i, s in enumerate(series_graficar):
            color = colores[i]
            
            # Dibujar línea fantasma
            ax.plot(x_total, s['serie'], label=s['label'], color=color, alpha=0.6, linewidth=1.2)
            
            # Dibujar Score al final de la línea (en el futuro)
            # Coordenadas: última vela del futuro (x) y valor final de la serie (y)
            ax.text(x_total[-1] + 1, s['serie'][-1], 
                    f"S:{s['score']:.2f}", 
                    color=color, fontsize=8, fontweight='bold', va='center')
            
        # Maestra
        y_master = np.insert(linea_maestra, 0, ultimo_valor_actual)
        x_master = np.insert(x_futuro, 0, 0)
        ax.plot(x_master, y_master, label="PROYECCIÓN PONDERADA", color='#00ff00', linewidth=4.0, zorder=10)
        
        # Actual
        ax.plot(x_pasado, patron_actual_norm, label=f"ACTUAL ({ticker_obj})", color='black', linewidth=2.5, zorder=11)
        
        # Realidad (Backtest)
        if enable_backtest and len(real_p) > 0:
            min_p = np.min(patron_actual)
            max_p = np.max(patron_actual)
            rng = max_p - min_p
            realidad_norm = (real_p - min_p) / rng
            
            limit_len = min(len(realidad_norm), proyeccion)
            y_real = realidad_norm[:limit_len]
            x_real = np.arange(1, limit_len + 1)
            y_real_con = np.insert(y_real, 0, ultimo_valor_actual)
            x_real_con = np.insert(x_real, 0, 0)
            
            ax.plot(x_real_con, y_real_con, label="REALIDAD (VALIDACIÓN)", color='black', linewidth=2.5, linestyle='--', zorder=12)

        # EJES
        ax.set_xlim(x_total[0], x_total[-1] + (proyeccion * 0.1)) # Margen derecho para etiquetas
        locator = ticker.MaxNLocator(nbins=25, integer=True)
        ax.xaxis.set_major_locator(locator)
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.3)
        ax.set_xlabel(f"Velas de {tf_obj}", fontsize=10, color='gray')
        
        # Eje Superior Tiempo
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
        ref_date = obj_f[-1]
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
        
        lbl_precio = f" Corte: {fecha_str} | ${curr_price:,.0f} " if enable_backtest else f" Actual: ${curr_price:,.0f} "
        ax2.text(x_pasado[0], curr_price, lbl_precio, 
                 color='black', fontweight='bold', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Layout
        plt.axvline(0, color='red', ls=':', label="Ahora")
        titulo = f"Fractalidad: {ticker_obj} ({tf_obj}) vs [{lib1_ticker} & {lib2_ticker}]"
        if enable_backtest: titulo += f" | BACKTEST: {fecha_str}"
        plt.title(titulo, pad=35, fontsize=14)
        
        # Leyenda fuera
        ax.legend(bbox_to_anchor=(1.08, 1), loc='upper left', fontsize=8, borderaxespad=0.)
        
        st.pyplot(fig)
        
        # Dataframe opcional
        with st.expander("Ver Datos de Proyección"):
            fechas_proy = [ref_date + (delta * i) for i in range(1, proyeccion + 1)]
            precios_proy = (linea_maestra * rng) + min_p
            st.dataframe(pd.DataFrame({"Fecha": fechas_proy, "Precio Estimado": precios_proy}))

    except Exception as e:
        st.error(f"Error: {e}")
