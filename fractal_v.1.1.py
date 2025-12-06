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

st.title("🔮 Radar de Fractales Multi-Mercado")
st.markdown("""
**Ingeniería Financiera:** Compara el patrón actual de tu activo contra **múltiples mercados históricos simultáneamente** para encontrar la mejor correlación estructural.
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

# --- HELPER UI ---
def render_asset_selector(label, key_prefix, default_val):
    col_mode, col_input = st.columns([1, 2])
    with col_mode:
        mode = st.radio("Modo", ["Lista", "Manual"], horizontal=True, key=f"{key_prefix}_mode", label_visibility="collapsed")
    with col_input:
        if mode == "Lista":
            idx = COMMON_TICKERS.index(default_val) if default_val in COMMON_TICKERS else 0
            val = st.selectbox(label, COMMON_TICKERS, index=idx, key=f"{key_prefix}_list")
        else:
            val = st.text_input(label, value=default_val, key=f"{key_prefix}_text")
    return val

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1. Objetivo (Patrón Actual)")
    ticker_obj = render_asset_selector("Activo", "target", "BTC-USD")
    
    c1, c2 = st.columns(2)
    with c1: tf_obj = st.selectbox("Timeframe", COMMON_TFS, index=4) 
    with c2: columna_analisis = st.selectbox("Precio:", ["Close", "Low", "High"], index=1) 

    st.divider()
    
    st.header("2. Librerías de Búsqueda")
    st.info("Selecciona contra qué activos comparar el pasado:")
    
    # MULTI-SELECTOR
    libs_seleccionadas = st.multiselect(
        "Activos de Referencia:", 
        options=COMMON_TICKERS,
        default=["QQQ", "GLD", "SPY", "NVDA"] # Default potente
    )
    
    # Opción para agregar uno manual extra a la búsqueda
    use_manual_lib = st.checkbox("Agregar activo manual extra")
    manual_lib = ""
    if use_manual_lib:
        manual_lib = st.text_input("Ticker Manual (Ej: ^IXIC)", value="")
        if manual_lib: libs_seleccionadas.append(manual_lib)

    tf_libs = st.selectbox("Timeframe de Búsqueda", ["1d", "1wk"], index=0)

    st.divider()
    
    st.header("3. Configuración")
    enable_backtest = st.checkbox("Activar Backtesting", value=False)
    fecha_corte = datetime.today()
    if enable_backtest:
        fecha_corte = st.date_input("Fecha Corte:", value=datetime.today()-timedelta(days=60), max_value=datetime.today())
    
    ventana = st.slider("Memoria (Velas)", 30, 365, 120)
    proyeccion = st.slider("Proyección (Futuro)", 5, 90, 30)
    resultados = st.slider("Top Coincidencias", 1, 15, 5)
    
    run_btn = st.button("🚀 EJECUTAR ESCÁNER", type="primary", use_container_width=True)

# --- LÓGICA CORE ---

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_y_procesar(ticker, tf, col_target, fecha_limite_str=None):
    tf_lower = tf.lower()
    if tf_lower == "1m": periodo = "7d"
    elif any(x in tf_lower for x in ["2m","5m","15m","30m","90m"]): periodo = "59d"
    elif any(x in tf_lower for x in ["60m","1h","4h"]): periodo = "730d"
    else: periodo = "max"
    
    try:
        data = yf.download(ticker, period=periodo, interval=tf, progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex): data = data[col_target]
        else: data = data[col_target]
        data = data.dropna()
        
        if len(data) < 50: return None, None, None, None, f"Pocos datos {ticker}"
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
    if not libs_seleccionadas:
        st.error("⚠️ Debes seleccionar al menos una librería de búsqueda.")
        st.stop()

    status_container = st.container()
    
    with status_container:
        st.info("📡 Iniciando descarga y procesamiento...")
        progreso = st.progress(0)

    try:
        fecha_str = fecha_corte.strftime('%Y-%m-%d') if enable_backtest else None
        
        # 1. OBJETIVO
        obj_p, obj_f, real_p, real_f, err = descargar_y_procesar(ticker_obj, tf_obj, columna_analisis, fecha_str)
        if err: st.error(err); st.stop()
        if len(obj_p) < ventana: st.error("Historial objetivo insuficiente."); st.stop()
        patron_actual = obj_p[-ventana:]
        
        progreso.progress(20)
        
        # 2. ESCANEO MULTI-LIBRERÍA (LOOP)
        matches_totales = []
        total_libs = len(libs_seleccionadas)
        
        for i, lib_ticker in enumerate(libs_seleccionadas):
            # Actualizar estado
            progreso.progress(20 + int((i / total_libs) * 50))
            
            lib_p, lib_f, _, _, err_lib = descargar_y_procesar(lib_ticker, tf_libs, columna_analisis, fecha_str)
            
            if err_lib:
                st.warning(f"Saltando {lib_ticker}: {err_lib}")
                continue
                
            # Escanear esta librería
            nuevos_matches = escanear_libreria(lib_ticker, lib_p, lib_f, patron_actual, proyeccion)
            matches_totales.extend(nuevos_matches)

        progreso.progress(75)
        
        if not matches_totales:
            st.error("No se encontraron coincidencias en ninguna librería.")
            st.stop()

        # 3. FILTRADO Y RANKING
        matches_totales.sort(key=lambda x: x['score'])
        
        seleccionados = []
        indices_usados = {} 
        distancia_min = int(ventana * 0.6)
        
        for m in matches_totales:
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
            
        progreso.progress(90)
        
        # 4. CONSTRUCCIÓN VISUAL
        suma_proyecciones = np.zeros(proyeccion)
        suma_pesos = 0
        patron_actual_norm = normalizar(patron_actual)
        ultimo_valor_actual = patron_actual_norm[-1]
        
        series_graficar = []
        for i, match in enumerate(seleccionados):
            rank = i + 1
            serie_comp = np.concatenate([match['datos_past'], match['datos_fut']])
            serie_norm = normalizar(serie_comp)
            punto_empalme = serie_norm[-(proyeccion+1)]
            offset = ultimo_valor_actual - punto_empalme
            serie_alineada = serie_norm + offset
            
            peso = 1 / (match['score']**2 + 0.0001)
            suma_proyecciones += serie_alineada[-proyeccion:] * peso
            suma_pesos += peso
            
            # Etiqueta limpia para leyenda
            label = f"#{rank} {match['source']} ({match['fecha_origen'].strftime('%Y-%m-%d')}) Score: {match['score']:.2f}"
            
            series_graficar.append({
                'serie': serie_alineada,
                'label': label,
                'score': match['score'],
                'rank': rank,
                'source': match['source']
            })
            
        linea_maestra = suma_proyecciones / suma_pesos
        progreso.progress(100)
        status_container.empty() # Limpiar barra de carga
        
        # --- PLOTEO ---
        fig, ax = plt.subplots(figsize=(16, 9))
        x_pasado = np.arange(-ventana + 1, 1)
        x_futuro = np.arange(1, proyeccion + 1)
        x_total = np.concatenate([x_pasado, x_futuro])
        
        # Mapa de colores dinámico (Un color único por rango)
        colores = plt.cm.tab10(np.linspace(0, 1, len(series_graficar)))
        
        # A. FANTASMAS
        for i, s in enumerate(series_graficar):
            color = colores[i]
            ax.plot(x_total, s['serie'], label=s['label'], color=color, alpha=0.6, linewidth=1.2)
            # Etiqueta numérica al final de la línea (#1, #2...)
            ax.text(x_total[-1] + (proyeccion * 0.02), s['serie'][-1], f"#{s['rank']}", 
                    color=color, fontsize=9, fontweight='bold', va='center')
            
        # B. MAESTRA
        y_master = np.insert(linea_maestra, 0, ultimo_valor_actual)
        x_master = np.insert(x_futuro, 0, 0)
        ax.plot(x_master, y_master, label="PROYECCIÓN PONDERADA", color='#00ff00', linewidth=4.0, zorder=10)
        
        # C. ACTUAL
        ax.plot(x_pasado, patron_actual_norm, label=f"ACTUAL ({ticker_obj})", color='black', linewidth=2.5, zorder=11)
        
        # D. REALIDAD
        if enable_backtest and len(real_p) > 0:
            min_p, max_p = np.min(patron_actual), np.max(patron_actual)
            rng = max_p - min_p
            realidad_norm = (real_p - min_p) / rng
            limit_len = min(len(realidad_norm), proyeccion)
            y_real = realidad_norm[:limit_len]
            x_real = np.arange(1, limit_len + 1)
            y_real_con = np.insert(y_real, 0, ultimo_valor_actual)
            x_real_con = np.insert(x_real, 0, 0)
            ax.plot(x_real_con, y_real_con, label="REALIDAD (VALIDACIÓN)", color='black', linewidth=2.5, linestyle='--', zorder=12)

        # CONFIGURACIÓN EJES
        ax.set_xlim(x_total[0], x_total[-1] + (proyeccion * 0.1))
        locator = ticker.MaxNLocator(nbins=25, integer=True)
        ax.xaxis.set_major_locator(locator)
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.3)
        ax.set_xlabel(f"Velas de {tf_obj}", fontsize=10, color='gray')
        
        # EJE TIEMPO SUPERIOR
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
        
        # EJE PRECIO
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
        
        label_v = "Fecha Corte" if enable_backtest else "Ahora"
        plt.axvline(0, color='red', ls=':', label=label_v)
        
        titulo_libs = ", ".join(libs_seleccionadas[:2]) + ("..." if len(libs_seleccionadas)>2 else "")
        titulo = f"Fractalidad: {ticker_obj} ({tf_obj}) vs [{titulo_libs}]"
        if enable_backtest: titulo += f" | BACKTEST: {fecha_str}"
        plt.title(titulo, pad=35, fontsize=14)
        
        # Leyenda fuera del gráfico para limpieza
        ax.legend(bbox_to_anchor=(1.08, 1), loc='upper left', fontsize=9, borderaxespad=0.)
        
        st.pyplot(fig)
        
        # TABLA OPCIONAL
        with st.expander("📊 Ver Proyección Numérica"):
            fechas_proy = [ref_date + (delta * i) for i in range(1, proyeccion + 1)]
            precios_proy = (linea_maestra * rng) + min_p
            st.dataframe(pd.DataFrame({"Fecha": fechas_proy, "Precio Estimado": precios_proy}))

    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")

