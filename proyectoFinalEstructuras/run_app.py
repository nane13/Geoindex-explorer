"""
Aplicación unificada GeoIndex Explorer.
"""
import importlib
import grid_module
importlib.reload(grid_module)
GridFile = grid_module.GridFile
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import sys, os, time
import colorsys
from folium.plugins import Draw
import json

# Asegurar que los módulos se importen desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from kd_tree_module import KDTree
from quadtree_modulo import Quadtree
from r_tree import RTree

# Ruta correcta del dataset (válida en local y en Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LOCAL_PATH = os.path.join(BASE_DIR, "data", "DatasetCoordenadas.csv")

# --------------------------
# Inicializar session_state
# --------------------------
if "ultimo_click" not in st.session_state:
    st.session_state.ultimo_click = None  # (lon, lat)

if "resultado_nn" not in st.session_state:
    st.session_state.resultado_nn = None  # dict {estructura, punto, distancia, nodos, tiempo}

if "resultado_rango" not in st.session_state:
    st.session_state.resultado_rango = None  # dict {estructura, caja, puntos, nodos, tiempo}

# --------------------------
# Configuración de la app
# --------------------------
st.set_page_config(layout="wide", page_title="GeoIndex Explorer")
st.title("🌍 GeoIndex Explorer — App")

st.write("Selecciona la estructura y realiza consultas.")

# --------------------------
# Menú lateral: selección
# --------------------------
st.sidebar.title("Selección y datos")
estructura = st.sidebar.selectbox("Estructura:", ("KD-Tree", "Quadtree", "Grid File", "R-Tree", "Comparación de estructuras"))

st.sidebar.markdown("---")
modo_carga = st.sidebar.radio("Fuente de datos:", ("Dataset interno ~300", "Subir CSV"))

df = None
if modo_carga == "Subir CSV":
    archivo = st.sidebar.file_uploader("Sube CSV (ciudad, latitud, longitud)", type=["csv", "txt"])
    if archivo is not None:
        # lectura robusta
        try:
            df = pd.read_csv(archivo)
        except:
            archivo.seek(0)
            df = pd.read_csv(archivo, header=None)
            if df.shape[1] == 1:
                df = df.iloc[:, 0].str.split(",", expand=True)
            if df.shape[1] == 3:
                df.columns = ["ciudad", "latitud", "longitud"]
        df.columns = [c.lower().strip() for c in df.columns]
        df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
        df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
        df = df.dropna().reset_index(drop=True)
else:
    # intenta cargar la copia local incluida; si no existe, intenta ruta en data/
    ruta = DATA_LOCAL_PATH if os.path.exists(DATA_LOCAL_PATH) else os.path.join("data", "DatasetCoordenadas.csv")
    df = pd.read_csv(ruta)

if df is None or df.empty:
    st.warning("Cargue datos (dataset interno o suba un CSV).")
    st.stop()

# vista previa
st.subheader("Vista previa de datos")
st.dataframe(df.head(20))

# preparar puntos (lon, lat, meta)
puntos = [(float(r["longitud"]), float(r["latitud"]), r["ciudad"]) for _, r in df.iterrows()]

# Mapa base (crear una sola vez por render)
lat_centro = float(df["latitud"].mean())
lon_centro = float(df["longitud"].mean())

def crear_mapa_base():
    m = folium.Map(location=[lat_centro, lon_centro], zoom_start=6, control_scale=True)
    # dibujar todos los puntos
    for lon, lat, meta in puntos:
        folium.CircleMarker(location=[lat, lon], radius=3, color="#3388ff", fill=True).add_to(m)
    return m

mapa = crear_mapa_base()
# ==============================
# KD-TREE
# ==============================
if estructura == "KD-Tree":

    # ----- construir KD-Tree -----
    t0 = time.time()
    kd = KDTree()
    kd.construir(puntos)
    tiempo_build = time.time() - t0

    st.subheader("KD-Tree — Métricas")
    col1, col2 = st.columns(2)
    col1.metric("Puntos", len(puntos))
    col2.metric("Tiempo construcción (s)", f"{tiempo_build:.6f}")

    # ----- TAB 1 -----
    tab_kd_nn, tab_kd_rango = st.tabs(["Vecino más cercano", "Consulta por rango"])

    with tab_kd_nn:
        st.write("Haz clic en el mapa para buscar automáticamente, o usa la consulta manual.")

        # ==============================
        # 1. Mostrar mapa base (sin resultados todavía)
        # ==============================
        mapa_data = st_folium(mapa, width=900, height=600)

        # Capturar clic
        last_clicked = mapa_data.get("last_clicked")
        resultado_automatico = None

        # ==============================
        # 2. Si el usuario hace CLIC ejecuta BÚSQUEDA AUTOMÁTICA
        # ==============================
        if last_clicked:
            lon_c = last_clicked["lng"]
            lat_c = last_clicked["lat"]

            mejor, dist_m, nodos_v, elapsed, recorrido  = kd.vecino_mas_cercano((lon_c, lat_c))

            if mejor:
                resultado_automatico = {
                    "consulta": (lon_c, lat_c),
                    "punto": mejor,
                    "dist": dist_m,
                    "nodos": nodos_v,
                    "tiempo": elapsed,
                    "recorrido": recorrido
                }

        # ==============================
        # 3. Consulta MANUAL con inputs
        # ==============================
        st.write("Consulta manual")

        # Inicializar keys
        if "resultado_manual" not in st.session_state:
            st.session_state.resultado_manual = None

        if "lat_manual" not in st.session_state:
            st.session_state.lat_manual = lat_centro

        if "lon_manual" not in st.session_state:
            st.session_state.lon_manual = lon_centro

        colA, colB = st.columns(2)

        lat_in = colA.number_input(
            "Latitud (manual)",
            key="lat_manual",
            format="%.6f"
        )
        lon_in = colB.number_input(
            "Longitud (manual)",
            key="lon_manual",
            format="%.6f"
        )

        # --------------------------
        # Botón de consulta manual
        # --------------------------
        if st.button("Buscar NN con coordenadas manuales"):
            consulta = (st.session_state.lon_manual, st.session_state.lat_manual)
            mejor, dist_m, nodos_v, elapsed, recorrido = kd.vecino_mas_cercano(consulta)

            if mejor:
                st.session_state.resultado_manual = {
                    "consulta": consulta,
                    "punto": mejor,
                    "dist": dist_m,
                    "nodos": nodos_v,
                    "tiempo": elapsed,
                    "recorrido": recorrido
                }

        # --------------------------
        # Mostrar resultado manual
        # --------------------------
        r = st.session_state.resultado_manual

        if r:
            st.subheader("Resultado (consulta manual)")
            st.write(f"Ciudad: **{r['punto'][2]}**")
            st.write(f"Distancia (m): {r['dist']:.1f} m")
            st.write(f"Nodos visitados: {r['nodos']}")
            st.write(f"Tiempo (s): {r['tiempo']:.6f}")

            # Mapa exclusivo del resultado manual
            mapa3 = crear_mapa_base()

            folium.Marker(
                location=[r["consulta"][1], r["consulta"][0]],
                popup="Consulta manual",
                icon=folium.Icon(color="green")
            ).add_to(mapa3)

            folium.Marker(
                location=[r["punto"][1], r["punto"][0]],
                popup=r["punto"][2],
                icon=folium.Icon(color="red")
            ).add_to(mapa3)

            folium.PolyLine(
                [[r["consulta"][1], r["consulta"][0]],
                [r["punto"][1], r["punto"][0]]],
                weight=2
            ).add_to(mapa3)

            #recorrido de los puntos
            n = len(r["recorrido"])
            for i, p in enumerate(r["recorrido"]):
                lon, lat = p[0], p[1]
                t = i / max(1, n - 1)  # de 0 a 1
                r, g, b = colorsys.hsv_to_rgb(0.12 - 0.12*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(r*255), int(g*255), int(b*255)
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color=color_hex,
                    fill=True,
                    fill_opacity=0.9
                ).add_to(mapa3)

            st_folium(mapa3, width=900, height=600)

        # ==============================
        # 4. Mostrar resultados AUTOMÁTICOS (clic)
        # ==============================
        if resultado_automatico:
            st.subheader("Resultado (clic en el mapa)")
            r = resultado_automatico

            st.write(f"Ciudad: **{r['punto'][2]}**")
            st.write(f"Distancia (m): {r['dist']:.1f}")
            st.write(f"Nodos visitados: {r['nodos']}")
            st.write(f"Tiempo (s): {r['tiempo']:.6f}")

            # Mapa final
            mapa2 = crear_mapa_base()

            # marcador de la consulta
            folium.Marker(
                location=[r["consulta"][1], r["consulta"][0]],
                popup="Consulta",
                icon=folium.Icon(color="green")
            ).add_to(mapa2)

            # marcador del NN
            folium.Marker(
                location=[r["punto"][1], r["punto"][0]],
                popup=r["punto"][2],
                icon=folium.Icon(color="red")
            ).add_to(mapa2)

            # línea
            folium.PolyLine(
                [
                    [r["consulta"][1], r["consulta"][0]],
                    [r["punto"][1], r["punto"][0]]
                ],
                weight=2
            ).add_to(mapa2)

            #recorrido de los puntos
            
            n = len(r["recorrido"])
            for i, p in enumerate(r["recorrido"]):
                lon, lat = p[0], p[1]
                t = i / max(1, n - 1)  # de 0 a 1
                r, g, b = colorsys.hsv_to_rgb(0.12 - 0.12*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(r*255), int(g*255), int(b*255)
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color=color_hex,
                    fill=True,
                    fill_opacity=0.9
                ).add_to(mapa2)

            st_folium(mapa2, width=900, height=600)
    # ---------------------------
    # TAB 2: Consulta por rango
    # ---------------------------
    with tab_kd_rango:

        st.write("Define un rectángulo y ejecuta una consulta por rango.")

        # -------------------------
        # Inicializar estado
        # -------------------------
        if "kd_rango_result" not in st.session_state:
            st.session_state.kd_rango_result = None

        # Inputs del rectángulo
        col1, col2 = st.columns(2)
        min_lon = col1.number_input("Longitud mínima", value=float(df["longitud"].min()), format="%.6f")
        max_lon = col2.number_input("Longitud máxima", value=float(df["longitud"].max()), format="%.6f")

        col3, col4 = st.columns(2)
        min_lat = col3.number_input("Latitud mínima", value=float(df["latitud"].min()), format="%.6f")
        max_lat = col4.number_input("Latitud máxima", value=float(df["latitud"].max()), format="%.6f")

        # -------------------------
        # Ejecutar consulta por rango
        # -------------------------
        if st.button("Ejecutar consulta por rango"):

            caja = (min_lon, max_lon, min_lat, max_lat)

            pts, nodos_vis, t_elapsed = kd.consulta_rango(caja)

            # Guardar persistente
            st.session_state.kd_rango_result = {
                "caja": caja,
                "pts": pts,
                "nodos": nodos_vis,
                "tiempo": t_elapsed
            }

        # -------------------------
        # Mostrar resultados si existen
        # -------------------------
        r = st.session_state.kd_rango_result
        if r:

            st.subheader("Resultados KD-Tree (consulta por rango)")
            st.write(f"**Puntos encontrados:** {len(r['pts'])}")
            st.write(f"**Nodos visitados:** {r['nodos']}")
            st.write(f"**Tiempo (s):** {r['tiempo']:.6f}")

            # Mapa persistente
            mapa_kd_rango = crear_mapa_base()

            min_lon, max_lon, min_lat, max_lat = r["caja"]

            # Dibujar rectángulo
            rect = [
                [min_lat, min_lon],
                [min_lat, max_lon],
                [max_lat, max_lon],
                [max_lat, min_lon],
                [min_lat, min_lon]
            ]
            folium.Polygon(
                rect,
                color="#1d3557",
                weight=2,
                fill=True,
                fill_opacity=0.05
            ).add_to(mapa_kd_rango)

            # Dibujar puntos encontrados
            for lon, lat, meta in r["pts"]:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color="#e63946",
                    fill=True
                ).add_to(mapa_kd_rango)

            st_folium(mapa_kd_rango, width=900, height=600)

elif estructura == "Quadtree":

    # ======================
    # Construcción Quadtree
    # ======================
    margen = 0.05
    min_lon = df["longitud"].min() - margen
    max_lon = df["longitud"].max() + margen
    min_lat = df["latitud"].min() - margen
    max_lat = df["latitud"].max() + margen
    limites = (min_lon, max_lon, min_lat, max_lat)

    cap = st.sidebar.number_input("Capacidad por nodo (QT)", min_value=1, max_value=100, value=8)
    maxd = st.sidebar.number_input("Profundidad máxima (QT)", min_value=1, max_value=50, value=20)

    t0 = time.time()
    qt = Quadtree(limites, cap, maxd)
    qt.construir(puntos)
    tiempo_build = time.time() - t0

    st.subheader("Quadtree — Métricas")
    est = qt.estadisticas()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Puntos", est["puntos_totales"])
    c2.metric("Nodos", est["nodos_totales"])
    c3.metric("Hojas", est["hojas"])
    c4.metric("Profundidad", est["profundidad_max"])
    st.write(f"Tiempo de construcción: {tiempo_build:.6f} s")

    # Dibujar subdivisiones
    mostrar_sub = st.sidebar.checkbox("Mostrar subdivisiones (QT)", value=True)
    if mostrar_sub:
        for n in qt._recolectar_nodos():
            color = "#d62828" if n.dividido else "#2a9d8f"
            caja = n.caja
            coords = [
                [caja[2], caja[0]],[caja[2], caja[1]],
                [caja[3], caja[1]],[caja[3], caja[0]],
                [caja[2], caja[0]]
            ]
            folium.Polygon(coords, color=color, weight=1, fill=False).add_to(mapa)

    # ======================
    # TABS
    # ======================
    tab_nn, tab_rango = st.tabs(["Vecino más cercano", "Consulta por rango"])

    # =====================================
    # TAB 1: VECINO MÁS CERCANO (QT)
    # =====================================
    with tab_nn:

        st.write("Haz clic en el mapa para buscar automáticamente o usa la consulta manual.")

        # ---- MOSTRAR MAPA BASE ----
        mapa_data = st_folium(mapa, width=900, height=600)
        last_clicked = mapa_data.get("last_clicked")

        # ---- RESULTADO AUTOMÁTICO ----
        resultado_auto_qt = None

        if last_clicked:
            lon_c = last_clicked["lng"]
            lat_c = last_clicked["lat"]

            mejor, dist_m, nodos_v, elapsed, recorrido= qt.vecino_mas_cercano((lon_c, lat_c))

            if mejor:
                resultado_auto_qt = {
                    "consulta": (lon_c, lat_c),
                    "punto": mejor,
                    "dist": dist_m,
                    "nodos": nodos_v,
                    "tiempo": elapsed,
                    "recorrido": recorrido
                }

        # -------------------------------
        # CONSULTA MANUAL CON SESSION_STATE
        # -------------------------------
        st.write("Consulta manual")

        # Inicializar keys
        if "resultado_manual_qt" not in st.session_state:
            st.session_state.resultado_manual_qt = None

        if "lat_manual_qt" not in st.session_state:
            st.session_state.lat_manual_qt = lat_centro
        if "lon_manual_qt" not in st.session_state:
            st.session_state.lon_manual_qt = lon_centro

        colA, colB = st.columns(2)

        lat_m = colA.number_input("Latitud (manual QT)", key="lat_manual_qt", format="%.6f")
        lon_m = colB.number_input("Longitud (manual QT)", key="lon_manual_qt", format="%.6f")

        if st.button("Buscar NN (manual) — QT"):
            consulta = (st.session_state.lon_manual_qt, st.session_state.lat_manual_qt)
            mejor, dist_m, nodos_v, elapsed, recorrido = qt.vecino_mas_cercano(consulta)

            if mejor:
                st.session_state.resultado_manual_qt = {
                    "consulta": consulta,
                    "punto": mejor,
                    "dist": dist_m,
                    "nodos": nodos_v,
                    "tiempo": elapsed,
                    "recorrido": recorrido
                }

        # -------------------------------
        # MOSTRAR RESULTADO AUTOMÁTICO
        # -------------------------------
        if resultado_auto_qt:
            st.subheader("Resultado (clic en el mapa) — Quadtree")
            r = resultado_auto_qt

            st.write(f"Ciudad: **{r['punto'][2]}**")
            st.write(f"Distancia (m): {r['dist']:.1f}")
            st.write(f"Nodos visitados: {r['nodos']}")
            st.write(f"Tiempo (s): {r['tiempo']:.6f}")

            mapa2 = crear_mapa_base()
            folium.Marker([r["consulta"][1], r["consulta"][0]], icon=folium.Icon(color="green")).add_to(mapa2)
            folium.Marker([r["punto"][1], r["punto"][0]], icon=folium.Icon(color="red")).add_to(mapa2)
            folium.PolyLine([[r["consulta"][1], r["consulta"][0]], [r["punto"][1], r["punto"][0]]], weight=2).add_to(mapa2)
            
            #recorrido de los puntos
            
            n = len(r["recorrido"])
            for i, p in enumerate(r["recorrido"]):
                lon, lat = p[0], p[1]
                t = i / max(1, n - 1)  # de 0 a 1
                r, g, b = colorsys.hsv_to_rgb(0.12 - 0.12*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(r*255), int(g*255), int(b*255)
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color=color_hex,
                    fill=True,
                    fill_opacity=0.9
                ).add_to(mapa2)
            
            st_folium(mapa2, width=900, height=600)

        # -------------------------------
        # MOSTRAR RESULTADO MANUAL (QT)
        # -------------------------------
        r = st.session_state.resultado_manual_qt

        if r:
            st.subheader("Resultado (consulta manual) — Quadtree")
            st.write(f"Ciudad: **{r['punto'][2]}**")
            st.write(f"Distancia (m): {r['dist']:.1f}")
            st.write(f"Nodos visitados: {r['nodos']}")
            st.write(f"Tiempo (s): {r['tiempo']:.6f}")

            mapa3 = crear_mapa_base()
            folium.Marker([r["consulta"][1], r["consulta"][0]], icon=folium.Icon(color="green")).add_to(mapa3)
            folium.Marker([r["punto"][1], r["punto"][0]], icon=folium.Icon(color="red")).add_to(mapa3)
            folium.PolyLine([[r["consulta"][1], r["consulta"][0]], [r["punto"][1], r["punto"][0]]], weight=2).add_to(mapa3)
            
            #recorrido de los puntos
            
            n = len(r["recorrido"])
            for i, p in enumerate(r["recorrido"]):
                lon, lat = p[0], p[1]
                t = i / max(1, n - 1)  # de 0 a 1
                r, g, b = colorsys.hsv_to_rgb(0.12 - 0.12*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(r*255), int(g*255), int(b*255)
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color=color_hex,
                    fill=True,
                    fill_opacity=0.9
                ).add_to(mapa3)

            st_folium(mapa3, width=900, height=600)

    # ---------- PESTAÑA RANGO ----------
    with tab_rango:
        st.write("Define un rectángulo para la consulta por rango.")
        min_lat = st.number_input("Latitud mínima", value=float(df["latitud"].min()))
        max_lat = st.number_input("Latitud máxima", value=float(df["latitud"].max()))
        min_lon = st.number_input("Longitud mínima", value=float(df["longitud"].min()))
        max_lon = st.number_input("Longitud máxima", value=float(df["longitud"].max()))

        if st.button("Ejecutar consulta por rango"):
            caja = (min_lon, max_lon, min_lat, max_lat)
            pts, nodos_v, t_elapsed = qt.consulta_rango(caja)
            st.session_state.resultado_rango = {
                "estructura": "Quadtree",
                "caja": caja, "puntos": pts, "nodos": nodos_v, "time": t_elapsed
            }
        # mostrar si hay resultado
        if st.session_state.resultado_rango and st.session_state.resultado_rango["estructura"] == "Quadtree":
            rr = st.session_state.resultado_rango
            st.subheader("Resultado - Rango")
            st.write(f"Puntos encontrados: {len(rr['puntos'])}")
            st.write(f"Nodos visitados: {rr['nodos']}")
            st.write(f"Tiempo (s): {rr['time']:.6f}")
            # dibujar rectángulo y puntos en mapa
            min_lon, max_lon, min_lat, max_lat = rr["caja"]
            rect_coords = [[min_lat, min_lon],[min_lat, max_lon],[max_lat, max_lon],[max_lat, min_lon],[min_lat, min_lon]]
            folium.Polygon(rect_coords, color="#ffa600", weight=2, fill=True, fill_opacity=0.05).add_to(mapa)
            for lon, lat, meta in rr["puntos"]:
                folium.CircleMarker(location=[lat, lon], radius=4, color="orange", fill=True).add_to(mapa)
            st_folium(mapa, width=900, height=600)

# --------------------------
# GRID FILE
# --------------------------
elif estructura == "Grid File":

    # construir GridFile
    limites = (df["longitud"].min()-0.01, df["longitud"].max()+0.01,
               df["latitud"].min()-0.01, df["latitud"].max()+0.01)

    filas = 20
    columnas = 20

    gf = GridFile(limites, rows=filas, cols=columnas)
    #PRUEBA#
    import inspect
    st.code(inspect.getsource(gf.vecino_mas_cercano))
    #PRUEBA#
    gf.construir(puntos)

    # Métricas
    st.subheader("Grid File — Métricas")
    stats = gf.estadisticas()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Filas x Columnas", f"{stats['rows']} x {stats['cols']}")
    c2.metric("Celdas totales", stats["celdas_totales"])
    c3.metric("Celdas usadas", stats["celdas_usadas"])
    c4.metric("Puntos (total)", stats["puntos_totales"])
    densidad = gf.factor_carga()
    st.metric("Factor de carga (densidad)", f"{densidad:.3f}")


    # Dibujo de celdas con color suave (opción B)
    # generamos colores alternados según (r+c) y opacidad baja
    pal = ["#F0F8FF", "#E6F2FF", "#FFF0F5", "#F5FFEE"]  # paleta suave
    mostrar_celdas = st.sidebar.checkbox("Mostrar celdas (Grid File)", value=True)
    if mostrar_celdas:
        for r in range(gf.rows):
            for c in range(gf.cols):
                bbox = gf._bbox_celda(r,c)
                min_lon, max_lon, min_lat, max_lat = bbox
                coords = [[min_lat, min_lon],[min_lat, max_lon],[max_lat, max_lon],[max_lat, min_lon],[min_lat, min_lon]]
                color = pal[(r+c) % len(pal)]
                folium.Polygon(coords, color=color, weight=1, fill=True, fill_opacity=0.12).add_to(mapa)

    # Tabs: NN y Rango
    tab_gn, tab_gr = st.tabs(["Vecino más cercano", "Consulta por rango"])

    # ---------- TAB NN ----------
    with tab_gn:
        st.write("Haz clic en el mapa para buscar automáticamente o usa la consulta manual.")

        # mapa para capturar clic
        mapa_data = st_folium(mapa, width=900, height=600)
        last_clicked = mapa_data.get("last_clicked")
        resultado_auto_gf = None

        if last_clicked:
            lon_c = last_clicked["lng"]
            lat_c = last_clicked["lat"]
            mejor, dist_m, celdas_v, t_elapsed, recorrido = gf.vecino_mas_cercano((lon_c, lat_c))
            if mejor:
                resultado_auto_gf = {
                    "consulta": (lon_c, lat_c),
                    "punto": mejor,
                    "dist": dist_m,
                    "celdas": celdas_v,
                    "tiempo": t_elapsed,
                    "recorrido": recorrido
                }

        # Consulta manual (session_state)
        if "resultado_manual_gf" not in st.session_state:
            st.session_state.resultado_manual_gf = None
        if "lat_manual_gf" not in st.session_state:
            st.session_state.lat_manual_gf = lat_centro
        if "lon_manual_gf" not in st.session_state:
            st.session_state.lon_manual_gf = lon_centro

        colA, colB = st.columns(2)
        lat_m = colA.number_input("Latitud (manual GF)", key="lat_manual_gf", format="%.6f")
        lon_m = colB.number_input("Longitud (manual GF)", key="lon_manual_gf", format="%.6f")

        if st.button("Buscar NN (manual) — Grid File"):
            consulta = (st.session_state.lon_manual_gf, st.session_state.lat_manual_gf)
            ##PRUEBAAAAAAAAAAAAAAA#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA#
            
            resultado = gf.vecino_mas_cercano(consulta)
            st.write("Resultado GF:", resultado, "len =", len(resultado))
            ##PRUEBAAAAAAAAAAAAAAA#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA#
            mejor, dist_m, celdas_v, t_elapsed, recorrido = gf.vecino_mas_cercano(consulta)
            if mejor:
                st.session_state.resultado_manual_gf = {
                    "consulta": consulta,
                    "punto": mejor,
                    "dist": dist_m,
                    "celdas": celdas_v,
                    "tiempo": t_elapsed,
                    "recorrido": recorrido 
                }

        # Mostrar resultado automático
        if resultado_auto_gf:
            r = resultado_auto_gf
            st.subheader("Resultado (clic) — Grid File")
            st.write(f"Ciudad: **{r['punto'][2]}**")
            st.write(f"Distancia (m): {r['dist']:.1f}")
            st.write(f"Celdas visitadas: {r['celdas']}")
            st.write(f"Tiempo (s): {r['tiempo']:.6f}")

            mapa2 = crear_mapa_base()

            folium.Marker(
                [r["consulta"][1], r["consulta"][0]],
                popup="Consulta",
                icon=folium.Icon(color="green")
            ).add_to(mapa2)

            folium.Marker(
                [r["punto"][1], r["punto"][0]],
                popup=r["punto"][2],
                icon=folium.Icon(color="red")
            ).add_to(mapa2)

            folium.PolyLine(
                [
                    [r["consulta"][1], r["consulta"][0]],
                    [r["punto"][1], r["punto"][0]]
                ],
                color="blue",
                weight=2
            ).add_to(mapa2)

            #  RECORRIDO DE CELDAS (DEGRADADO)
            # -----------------------------
            n = len(r["recorrido"])
            for i, (fila, col) in enumerate(r["recorrido"]):

                # color degradado verde → rojo
                t = i / max(1, n - 1)
                rr, gg, bb = colorsys.hsv_to_rgb(0.27 - 0.27*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(int(rr*255), int(gg*255), int(bb*255))

                # obtener bbox de la celda
                bbox = gf._bbox_celda(fila, col)
                lon_min, lon_max, lat_min, lat_max = bbox

                # centro de la celda
                lon_c = (lon_min + lon_max) / 2
                lat_c = (lat_min + lat_max) / 2

                folium.CircleMarker(
                    location=[lat_c, lon_c],
                    radius=5,
                    color=color_hex,
                    fill=True,
                    fill_opacity=0.9
                ).add_to(mapa2)            

            st_folium(mapa2, width=900, height=600)

        # Mostrar resultado manual persistente
        rm = st.session_state.resultado_manual_gf
        if rm:
            st.subheader("Resultado (manual) — Grid File")
            st.write(f"Ciudad: **{rm['punto'][2]}**")
            st.write(f"Distancia (m): {rm['dist']:.1f}")
            st.write(f"Celdas visitadas: {rm['celdas']}")
            st.write(f"Tiempo (s): {rm['tiempo']:.6f}")

            mapa3 = crear_mapa_base()
            folium.Marker([rm["consulta"][1], rm["consulta"][0]], popup="Consulta manual", icon=folium.Icon(color="green")).add_to(mapa3)
            folium.Marker([rm["punto"][1], rm["punto"][0]], popup=rm["punto"][2], icon=folium.Icon(color="red")).add_to(mapa3)
            folium.PolyLine([[rm["consulta"][1], rm["consulta"][0]],[rm["punto"][1], rm["punto"][0]]], weight=2).add_to(mapa3)
            
            #  RECORRIDO DE CELDAS (DEGRADADO)
            # -----------------------------
            n = len(rm["recorrido"])
            for i, (fila, col) in enumerate(rm["recorrido"]):

                # color degradado verde → rojo
                t = i / max(1, n - 1)
                rr, gg, bb = colorsys.hsv_to_rgb(0.27 - 0.27*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(int(rr*255), int(gg*255), int(bb*255))

                # obtener bbox de la celda
                bbox = gf._bbox_celda(fila, col)
                lon_min, lon_max, lat_min, lat_max = bbox

                # centro de la celda
                lon_c = (lon_min + lon_max) / 2
                lat_c = (lat_min + lat_max) / 2

                folium.CircleMarker(
                    location=[lat_c, lon_c],
                    radius=5,
                    color=color_hex,
                    fill=True,
                    fill_opacity=0.9
                ).add_to(mapa3)

            st_folium(mapa3, width=900, height=600)

    # ---------- TAB RANGO ----------
    if "resultado_rango_grid" not in st.session_state:
        st.session_state.resultado_rango_grid = None

    with tab_gr:
        st.write("Consulta por rango (rectángulo)")

        # Inputs de rango
        min_lat = st.number_input("Latitud mínima", value=float(df["latitud"].min()))
        max_lat = st.number_input("Latitud máxima", value=float(df["latitud"].max()))
        min_lon = st.number_input("Longitud mínima", value=float(df["longitud"].min()))
        max_lon = st.number_input("Longitud máxima", value=float(df["longitud"].max()))

        # Ejecutar consulta
        if st.button("Ejecutar rango (Grid File)"):
            caja = (min_lon, max_lon, min_lat, max_lat)
            pts, visitadas, t_elapsed = gf.consulta_rango(caja)

            # Guardar en session_state
            st.session_state.resultado_rango_grid = {
                "caja": caja,
                "pts": pts,
                "visitadas": visitadas,
                "tiempo": t_elapsed
            }

        # Mostrar resultado persistente
        if st.session_state.resultado_rango_grid:
            r = st.session_state.resultado_rango_grid

            st.subheader("Resultado rango (Grid File)")
            st.write(f"**Puntos encontrados:** {len(r['pts'])}")
            st.write(f"**Celdas visitadas:** {r['visitadas']}")
            st.write(f"**Tiempo (s):** {r['tiempo']:.6f}")

            # Dibujar mapa con resultados
            mapa_r = crear_mapa_base()
            min_lon, max_lon, min_lat, max_lat = r["caja"]

            # Rectángulo de consulta
            rect = [
                [min_lat, min_lon],
                [min_lat, max_lon],
                [max_lat, max_lon],
                [max_lat, min_lon],
                [min_lat, min_lon],
            ]
            folium.Polygon(
                rect,
                color="#ffa600",
                weight=2,
                fill=True,
                fill_opacity=0.05
            ).add_to(mapa_r)

            # Puntos encontrados
            for lon, lat, meta in r["pts"]:
                folium.CircleMarker(
                    [lat, lon],
                    radius=4,
                    color="orange",
                    fill=True
                ).add_to(mapa_r)

            st_folium(mapa_r, width=900, height=600)

# --------------------------
# R-TREE (Guttman - split LINEAL)
# --------------------------
elif estructura == "R-Tree":

    # Parámetros
    max_entries = st.sidebar.number_input("Max entradas por nodo (R-Tree)", min_value=2, max_value=64, value=8)
    t0 = time.time()
    rt = RTree(max_entries=max_entries)
    # insertar puntos
    rt_insert_start = time.time()
    for p in puntos:
        rt.insertar(p)
    tiempo_build = time.time() - t0

    st.subheader("R-Tree — Métricas")
    c1, c2, c3 = st.columns(3)
    c1.metric("Puntos", rt.n_puntos)
    c2.metric("Tiempo construcción (s)", f"{tiempo_build:.6f}")
    c3.metric("Max entradas", max_entries)
    densidad = rt.factor_carga()
    st.metric("Factor de carga (densidad)", f"{densidad:.3f}")

    # Mostrar MBRs del árbol (opcional)
    mostrar_mbrs = st.sidebar.checkbox("Mostrar MBRs (R-Tree)", value=True)
    if mostrar_mbrs:
        for c in rt.mbrs_a_dibujar():
            coords = [[c[2], c[0]],[c[2], c[1]],[c[3], c[1]],[c[3], c[0]],[c[2], c[0]]]
            folium.Polygon(coords, color="#6a4c93", weight=1, fill=False).add_to(mapa)

    # TABS: NN y Rango
    tab_rt_nn, tab_rt_r= st.tabs(["Vecino más cercano", "Consulta por rango"])

    # ---- TAB NN ----
    with tab_rt_nn:
        st.write("Haz clic en el mapa para buscar automáticamente o usa la consulta manual.")
        data = st_folium(mapa, width=900, height=600)
        last_clicked = data.get("last_clicked")

        resultado_auto_rt = None
        if last_clicked:
            lon_c = last_clicked["lng"]
            lat_c = last_clicked["lat"]
            mejor, dist_m, visitas, elapsed, recorrido = rt.vecino_mas_cercano((lon_c, lat_c))
            if mejor:
                resultado_auto_rt = {
                    "consulta": (lon_c, lat_c),
                    "punto": mejor,
                    "dist": dist_m,
                    "visitadas": visitas,
                    "tiempo": elapsed,
                    "recorrido": recorrido
                }

        # Consulta manual (session_state)
        if "resultado_manual_rt" not in st.session_state:
            st.session_state.resultado_manual_rt = None
        if "lat_manual_rt" not in st.session_state:
            st.session_state.lat_manual_rt = lat_centro
        if "lon_manual_rt" not in st.session_state:
            st.session_state.lon_manual_rt = lon_centro

        colA, colB = st.columns(2)
        lat_m = colA.number_input("Latitud (manual RT)", key="lat_manual_rt", format="%.6f")
        lon_m = colB.number_input("Longitud (manual RT)", key="lon_manual_rt", format="%.6f")

        if st.button("Buscar NN (manual) — R-Tree"):
            consulta = (st.session_state.lon_manual_rt, st.session_state.lat_manual_rt)
            mejor, dist_m, visitas, elapsed, recorrido = rt.vecino_mas_cercano(consulta)
            if mejor:
                st.session_state.resultado_manual_rt = {
                    "consulta": consulta,
                    "punto": mejor,
                    "dist": dist_m,
                    "visitadas": visitas,
                    "tiempo": elapsed,
                    "recorrido": recorrido
                }

        # Mostrar resultado automático
        if resultado_auto_rt:
            r = resultado_auto_rt
            st.subheader("Resultado (clic) — R-Tree")
            st.write(f"Ciudad: **{r['punto'][2]}**")
            st.write(f"Distancia (m): {r['dist']:.1f}")
            st.write(f"Nodos visitados: {r['visitadas']}")
            st.write(f"Tiempo (s): {r['tiempo']:.6f}")

            mapa2 = crear_mapa_base()
            folium.Marker([r["consulta"][1], r["consulta"][0]], icon=folium.Icon(color="green")).add_to(mapa2)
            folium.Marker([r["punto"][1], r["punto"][0]], icon=folium.Icon(color="red")).add_to(mapa2)
            folium.PolyLine(
                [
                    [r["consulta"][1], r["consulta"][0]],
                    [r["punto"][1], r["punto"][0]]
                ],
                weight=2
            ).add_to(mapa2)

            # DIBUJAR MBRs RECORRIDOS CON DEGRADADO
            rec = r["recorrido"]
            n = len(rec)
            for i, (min_lon, max_lon, min_lat, max_lat) in enumerate(rec):

                t = i / max(1, n - 1)
                rr, gg, bb = colorsys.hsv_to_rgb(0.27 - 0.27*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(int(rr*255), int(gg*255), int(bb*255))

                coords = [
                    [min_lat, min_lon],
                    [min_lat, max_lon],
                    [max_lat, max_lon],
                    [max_lat, min_lon],
                    [min_lat, min_lon]
                ]

                folium.Polygon(coords, color=color_hex, weight=2, fill=False).add_to(mapa2)
            st_folium(mapa2, width=900, height=600)

        # Mostrar resultado manual persistente
        rm = st.session_state.resultado_manual_rt
        if rm:
            st.subheader("Resultado (manual) — R-Tree")
            st.write(f"Ciudad: **{rm['punto'][2]}**")
            st.write(f"Distancia (m): {rm['dist']:.1f}")
            st.write(f"Nodos visitados: {rm['visitadas']}")
            st.write(f"Tiempo (s): {rm['tiempo']:.6f}")

            mapa3 = crear_mapa_base()
            folium.Marker([rm["consulta"][1], rm["consulta"][0]], icon=folium.Icon(color="green")).add_to(mapa3)
            folium.Marker([rm["punto"][1], rm["punto"][0]], icon=folium.Icon(color="red")).add_to(mapa3)
            folium.PolyLine(
                [
                    [rm["consulta"][1], rm["consulta"][0]],
                    [rm["punto"][1], rm["punto"][0]]
                ],
                weight=2
            ).add_to(mapa3)

            # DIBUJAR MBRs RECORRIDOS CON DEGRADADO
            rec = rm["recorrido"]
            n = len(rec)
            for i, (min_lon, max_lon, min_lat, max_lat) in enumerate(rec):

                t = i / max(1, n - 1)
                rr, gg, bb = colorsys.hsv_to_rgb(0.27 - 0.27*t, 1, 1)
                color_hex = "#{:02x}{:02x}{:02x}".format(int(rr*255), int(gg*255), int(bb*255))

                coords = [
                    [min_lat, min_lon],
                    [min_lat, max_lon],
                    [max_lat, max_lon],
                    [max_lat, min_lon],
                    [min_lat, min_lon]
                ]

                folium.Polygon(coords, color=color_hex, weight=2, fill=False).add_to(mapa3)
            st_folium(mapa3, width=900, height=600)

    # ---- TAB RANGO ----
    with tab_rt_r:
        st.write("Consulta por rango (rectángulo)")
        min_lat = st.number_input("Lat min (RT)", value=float(df["latitud"].min()))
        max_lat = st.number_input("Lat max (RT)", value=float(df["latitud"].max()))
        min_lon = st.number_input("Lon min (RT)", value=float(df["longitud"].min()))
        max_lon = st.number_input("Lon max (RT)", value=float(df["longitud"].max()))

        if st.button("Ejecutar rango (R-Tree)"):
            caja = (min_lon, max_lon, min_lat, max_lat)
            pts, visited, t_elapsed = rt.buscar_rango(caja)
            st.session_state.resultado_rango_rt = {
                "caja": caja, "pts": pts, "visited": visited, "tiempo": t_elapsed
            }

        if st.session_state.get("resultado_rango_rt"):
            rr = st.session_state.resultado_rango_rt
            st.subheader("Resultado rango (R-Tree)")
            st.write(f"Puntos encontrados: {len(rr['pts'])}")
            st.write(f"Nodos visitados: {rr['visited']}")
            st.write(f"Tiempo (s): {rr['tiempo']:.6f}")

            mapa_r = crear_mapa_base()
            min_lon, max_lon, min_lat, max_lat = rr["caja"]
            rect = [[min_lat, min_lon],[min_lat, max_lon],[max_lat, max_lon],[max_lat, min_lon],[min_lat, min_lon]]
            folium.Polygon(rect, color="#ffa600", weight=2, fill=True, fill_opacity=0.05).add_to(mapa_r)
            for lon, lat, meta in rr["pts"]:
                folium.CircleMarker([lat, lon], radius=4, color="orange", fill=True).add_to(mapa_r)
            st_folium(mapa_r, width=900, height=600)
   
# --------------------------
# Comparación de estructuras (módulo integrado)
# --------------------------
elif estructura == "Comparación de estructuras":
    import io
    import matplotlib.pyplot as plt
    import pandas as pd

    # Construcción de estructuras para el benchmark
    # KD-Tree
    kd = KDTree()
    kd.construir(puntos)

    # Quadtree
    limites = (df["longitud"].min(), df["longitud"].max(),
            df["latitud"].min(), df["latitud"].max())
    qt = Quadtree(limites, 8, 20)
    qt.construir(puntos)

    # Grid File
    gf_limites = limites
    gf = GridFile(gf_limites, rows=20, cols=20)
    gf.construir(puntos)

    # R-Tree
    rt = RTree(max_entries=5)
    for p in puntos:
        rt.insertar(p)

    # Colores por estructura (para dibujar en el mapa)
    colores = {
        "KD-Tree": "#e63946",
        "Quadtree": "#457b9d",
        "Grid File": "#f4a261",
        "R-Tree": "#8d6cab"
    }

    # estado persistente
    if "bench_results" not in st.session_state:
        st.session_state.bench_results = {"nn_click": None, "nn_manual": None, "rango": None}
    if "bench_last_point" not in st.session_state:
        st.session_state.bench_last_point = None

    st.header("Comparación de estructuras")
    st.write("Selecciona un punto en el mapa o ingresa las coordenadas para obtener la comparación entre las estructuras.")

    # Asegurar que las estructuras existen (si no, construir con parámetros razonables)
    try:
        _ = kd
    except Exception:
        try:
            from kd_tree_module import KDTree
            kd = KDTree()
            kd.construir(puntos)
        except Exception:
            kd = None

    try:
        _ = qt
    except Exception:
        try:
            from quadtree_modulo import Quadtree
            limites_def = (df["longitud"].min()-0.01, df["longitud"].max()+0.01,
                           df["latitud"].min()-0.01, df["latitud"].max()+0.01)
            qt = Quadtree(limites_def, 8, 20)
            qt.construir(puntos)
        except Exception:
            qt = None

    try:
        _ = gf
    except Exception:
        try:
            from grid_module import GridFile
            limites_def = (df["longitud"].min()-0.01, df["longitud"].max()+0.01,
                           df["latitud"].min()-0.01, df["latitud"].max()+0.01)
            gf = GridFile(limites_def, rows=20, cols=20)
            gf.construir(puntos)
        except Exception:
            gf = None

    try:
        _ = rt
    except Exception:
        try:
            from r_tree import RTree
            rt = RTree(max_entries=8)
            for p in puntos:
                rt.insertar(p)
        except Exception:
            rt = None

    # mapa fijo del módulo
    mapa_comp = crear_mapa_base()
    mapa_data = st_folium(mapa_comp, width=900, height=420)
    last_click_cmp = mapa_data.get("last_clicked")
    if last_click_cmp:
        st.session_state.bench_last_point = (last_click_cmp["lng"], last_click_cmp["lat"])

    # Tabs: NN-clic, NN-manual, Rango
    tab_click, tab_manual, tab_rango = st.tabs(["NN — por clic", "NN — manual", "Consulta por rango"])

    # ---------------------
    # TAB 1: NN — por clic (auto)
    # ---------------------
    with tab_click:
        st.write("Haz clic en el mapa (arriba). Al detectar un clic, la app ejecuta automáticamente la comparativa NN en todas las estructuras y dibuja los resultados.")
        if st.session_state.bench_last_point:
            punto = st.session_state.bench_last_point
            st.info(f"Punto seleccionado: lon={punto[0]:.6f}, lat={punto[1]:.6f} — ejecutando benchmark...")
            # Ejecutar comparativa NN automáticamente
            results_nn = []

            # KD-Tree
            try:
                if kd is not None:
                    mejor, dist, nodes_v, t_elapsed, recorrido = kd.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"KD-Tree","dist_m":dist,"nodos_visitados":nodes_v,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"KD-Tree","error":"KD no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"KD-Tree","error":str(e)})

            # Quadtree
            try:
                if qt is not None:
                    mejor, dist, nodes_v, t_elapsed, recorrido = qt.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"Quadtree","dist_m":dist,"nodos_visitados":nodes_v,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"Quadtree","error":"QT no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"Quadtree","error":str(e)})

            # Grid File
            try:
                if gf is not None:
                    mejor, dist, cells_v, t_elapsed, recorrido = gf.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"Grid File","dist_m":dist,"nodos_visitados":cells_v,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"Grid File","error":"GF no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"Grid File","error":str(e)})

            # R-Tree
            try:
                if rt is not None:
                    mejor, dist, visited, t_elapsed, recorrido = rt.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"R-Tree","dist_m":dist,"nodos_visitados":visited,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"R-Tree","error":"RT no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"R-Tree","error":str(e)})

            st.session_state.bench_results["nn_click"] = {"punto": punto, "results": results_nn}

        # mostrar si hay resultados previos
        if st.session_state.bench_results.get("nn_click"):
            b = st.session_state.bench_results["nn_click"]
            punto_cons = b["punto"]
            df_nn = pd.DataFrame(b["results"])
            st.subheader("Resultados NN (clic)")
            st.dataframe(df_nn)

            # mapa de resultados
            mapa_rn = crear_mapa_base()
            folium.Marker([punto_cons[1], punto_cons[0]], popup="Consulta (clic)", icon=folium.Icon(color="green")).add_to(mapa_rn)
            for row in b["results"]:
                if "error" in row:
                    continue
                est = row["estructura"]
                try:
                    if est == "KD-Tree" and kd is not None:
                        mejor, _, _, _, _ = kd.vecino_mas_cercano(punto_cons)
                    elif est == "Quadtree" and qt is not None:
                        mejor, _, _, _, _ = qt.vecino_mas_cercano(punto_cons)
                    elif est == "Grid File" and gf is not None:
                        mejor, _, _, _, _ = gf.vecino_mas_cercano(punto_cons)
                    elif est == "R-Tree" and rt is not None:
                        mejor, _, _, _, _ = rt.vecino_mas_cercano(punto_cons)
                    else:
                        mejor = None
                except Exception:
                    mejor = None

                if mejor:
                    nombre = mejor[2] if len(mejor) > 2 else est
                    color = colores.get(est, "#444444")
                    folium.Marker([mejor[1], mejor[0]], popup=f"{est}: {nombre}", icon=folium.Icon(color="red")).add_to(mapa_rn)
                    folium.PolyLine([[punto_cons[1], punto_cons[0]],[mejor[1], mejor[0]]], weight=2, color=color).add_to(mapa_rn)

            st_folium(mapa_rn, width=900, height=600)

            # graficas
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            try:
                df_plot = df_nn.dropna(subset=["tiempo_s"]).set_index("estructura")
                df_plot["tiempo_s"].plot.bar(ax=ax[0], title="Tiempo NN (s)")
            except Exception:
                ax[0].text(0.5,0.5,"No hay datos de tiempo", ha="center")
            try:
                df_plot2 = df_nn.dropna(subset=["nodos_visitados"]).set_index("estructura")
                df_plot2["nodos_visitados"].plot.bar(ax=ax[1], title="Nodos/Celdas visitados")
            except Exception:
                ax[1].text(0.5,0.5,"No hay datos de nodos", ha="center")
            plt.tight_layout()
            st.pyplot(fig)

            # ---- Calcular densidades ----

            # Grid File
            num_celdas = gf.rows * gf.cols
            dens_gf = len(puntos) / num_celdas

            # R-Tree
            def contar_mbrs_rtree(rt):
                nodos = []
                def rec(node):
                    nodos.append(node)
                    if not node.hoja:
                        for e in node.entries:
                            if e.hijo:
                                rec(e.hijo)
                rec(rt.root)
                return sum(len(n.entries) for n in nodos)

            total_mbrs = contar_mbrs_rtree(rt)
            dens_rt = len(puntos) / total_mbrs

            fig2, ax = plt.subplots(figsize=(3,2))
            df_dens = pd.DataFrame({
                "estructura": ["Grid File", "R-Tree"],
                "densidad": [dens_gf, dens_rt]
            }).set_index("estructura")

            df_dens["densidad"].plot.bar(ax=ax, title="Factor de densidad (GF vs R-Tree)")
            st.pyplot(fig2, use_container_width=False)

            # descargar csv
            csv_buf = io.StringIO()
            df_nn.to_csv(csv_buf, index=False)
            st.download_button("Descargar CSV (NN - clic)", data=csv_buf.getvalue().encode("utf-8"), file_name="bench_nn_click.csv", mime="text/csv")

    # ---------------------
    # TAB 2: NN — manual (botón)
    # ---------------------
    with tab_manual:
        st.write("Introduce coordenadas y pulsa el botón para ejecutar la comparativa NN en las 4 estructuras.")
        colA, colB = st.columns(2)
        lat_m = colA.number_input("Latitud (manual NN)", value=float(lat_centro), format="%.6f", key="bench_lat_manual")
        lon_m = colB.number_input("Longitud (manual NN)", value=float(lon_centro), format="%.6f", key="bench_lon_manual")
        
        # Inicializar contenedores SIEMPRE
        results_nn = []
        results_rango = []

        if st.button("Ejecutar benchmark NN (manual)"):
            punto = (lon_m, lat_m)
            results_nn = []

            try:
                if kd is not None:
                    mejor, dist, nodes_v, t_elapsed, recorrido = kd.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"KD-Tree","dist_m":dist,"nodos_visitados":nodes_v,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"KD-Tree","error":"KD no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"KD-Tree","error":str(e)})

            try:
                if qt is not None:
                    mejor, dist, nodes_v, t_elapsed, recorrido = qt.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"Quadtree","dist_m":dist,"nodos_visitados":nodes_v,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"Quadtree","error":"QT no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"Quadtree","error":str(e)})

            try:
                if gf is not None:
                    mejor, dist, cells_v, t_elapsed, recorrido = gf.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"Grid File","dist_m":dist,"nodos_visitados":cells_v,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"Grid File","error":"GF no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"Grid File","error":str(e)})

            try:
                if rt is not None:
                    mejor, dist, visited, t_elapsed, recorrido = rt.vecino_mas_cercano(punto)
                    results_nn.append({"estructura":"R-Tree","dist_m":dist,"nodos_visitados":visited,"tiempo_s":t_elapsed})
                else:
                    results_nn.append({"estructura":"R-Tree","error":"RT no disponible"})
            except Exception as e:
                results_nn.append({"estructura":"R-Tree","error":str(e)})

            st.session_state.bench_results["nn_manual"] = {"punto": punto, "results": results_nn}

        # mostrar si hay resultado
        if st.session_state.bench_results.get("nn_manual"):
            b = st.session_state.bench_results["nn_manual"]
            punto_cons = b["punto"]
            df_nn = pd.DataFrame(b["results"])
            st.subheader("Resultados NN (manual)")
            st.dataframe(df_nn)

            mapa_rn = crear_mapa_base()
            folium.Marker([punto_cons[1], punto_cons[0]], popup="Consulta (manual)", icon=folium.Icon(color="green")).add_to(mapa_rn)
            for row in b["results"]:
                if "error" in row:
                    continue
                est = row["estructura"]
                try:
                    if est == "KD-Tree" and kd is not None:
                        mejor, _, _, _, _ = kd.vecino_mas_cercano(punto_cons)
                    elif est == "Quadtree" and qt is not None:
                        mejor, _, _, _, _ = qt.vecino_mas_cercano(punto_cons)
                    elif est == "Grid File" and gf is not None:
                        mejor, _, _, _, _ = gf.vecino_mas_cercano(punto_cons)
                    elif est == "R-Tree" and rt is not None:
                        mejor, _, _, _, _ = rt.vecino_mas_cercano(punto_cons)
                    else:
                        mejor = None
                except Exception:
                    mejor = None

                if mejor:
                    nombre = mejor[2] if len(mejor) > 2 else est
                    color = colores.get(est, "#444444")
                    folium.Marker([mejor[1], mejor[0]], popup=f"{est}: {nombre}", icon=folium.Icon(color="red")).add_to(mapa_rn)
                    folium.PolyLine([[punto_cons[1], punto_cons[0]],[mejor[1], mejor[0]]], weight=2, color=color).add_to(mapa_rn)

            st_folium(mapa_rn, width=900, height=600)

            fig, ax = plt.subplots(1,2, figsize=(10,4))
            try:
                df_plot = df_nn.dropna(subset=["tiempo_s"]).set_index("estructura")
                df_plot["tiempo_s"].plot.bar(ax=ax[0], title="Tiempo NN (s)")
            except Exception:
                ax[0].text(0.5,0.5,"No hay datos de tiempo", ha="center")
            try:
                df_plot2 = df_nn.dropna(subset=["nodos_visitados"]).set_index("estructura")
                df_plot2["nodos_visitados"].plot.bar(ax=ax[1], title="Nodos/Celdas visitados")
            except Exception:
                ax[1].text(0.5,0.5,"No hay datos de nodos", ha="center")
            plt.tight_layout()
            st.pyplot(fig)

            # ---- Calcular densidades ----

            # Grid File
            num_celdas = gf.rows * gf.cols
            dens_gf = len(puntos) / num_celdas

            # R-Tree
            def contar_mbrs_rtree(rt):
                nodos = []
                def rec(node):
                    nodos.append(node)
                    if not node.hoja:
                        for e in node.entries:
                            if e.hijo:
                                rec(e.hijo)
                rec(rt.root)
                return sum(len(n.entries) for n in nodos)

            total_mbrs = contar_mbrs_rtree(rt)
            dens_rt = len(puntos) / total_mbrs

            fig2, ax = plt.subplots(figsize=(3,2))
            df_dens = pd.DataFrame({
                "estructura": ["Grid File", "R-Tree"],
                "densidad": [dens_gf, dens_rt]
            }).set_index("estructura")

            df_dens["densidad"].plot.bar(ax=ax, title="Factor de densidad (GF vs R-Tree)")
            st.pyplot(fig2, use_container_width=False)

            csv_buf = io.StringIO()
            df_nn.to_csv(csv_buf, index=False)
            st.download_button("Descargar CSV (NN - manual)", data=csv_buf.getvalue().encode("utf-8"), file_name="bench_nn_manual.csv", mime="text/csv")

    # ---------------------
    # TAB 3: Consulta por rango
    # ---------------------
    with tab_rango:
        st.write("Define el rectángulo y pulsa el botón para ejecutar la comparativa por rango.")

        min_lat_r = st.number_input("Lat min (rango)", value=float(df["latitud"].min()), key="cmp_min_lat")
        max_lat_r = st.number_input("Lat max (rango)", value=float(df["latitud"].max()), key="cmp_max_lat")
        min_lon_r = st.number_input("Lon min (rango)", value=float(df["longitud"].min()), key="cmp_min_lon")
        max_lon_r = st.number_input("Lon max (rango)", value=float(df["longitud"].max()), key="cmp_max_lon")

        # -------------------------------
        # 1) EJECUTAR Y GUARDAR RESULTADOS
        # -------------------------------
        if st.button("Ejecutar benchmark Rango"):

            caja = (min_lon_r, max_lon_r, min_lat_r, max_lat_r)
            results_rango = []

            # KD-Tree
            try:
                if kd is not None and hasattr(kd, "consulta_rango"):
                    pts, nodos_v, t_elapsed = kd.consulta_rango(caja)

                    results_rango.append({
                        "estructura": "KD-Tree",
                        "puntos_encontrados": len(pts),
                        "nodos_visitados": nodos_v,
                        "tiempo_s": t_elapsed
                    })

                else:
                    # fallback si kd no tiene el método (no debería ocurrir)
                    t0 = time.perf_counter()
                    pts = [
                        p for p in puntos
                        if (caja[0] <= p[0] <= caja[1] and caja[2] <= p[1] <= caja[3])
                    ]
                    t_elapsed = time.perf_counter() - t0

                    results_rango.append({
                        "estructura": "KD-Tree (scan fallback)",
                        "puntos_encontrados": len(pts),
                        "nodos_visitados": len(puntos),
                        "tiempo_s": t_elapsed
                    })

            except Exception as e:
                results_rango.append({
                    "estructura": "KD-Tree",
                    "error": str(e)
                })
            # Quadtree
            try:
                pts, nodes_v, t_elapsed = qt.consulta_rango(caja)
                results_rango.append({
                    "estructura":"Quadtree",
                    "puntos_encontrados":len(pts),
                    "nodos_visitados":nodes_v,
                    "tiempo_s":t_elapsed
                })
            except Exception as e:
                results_rango.append({"estructura":"Quadtree","error":str(e)})

            # Grid File
            try:
                pts, cells_v, t_elapsed = gf.consulta_rango(caja)
                results_rango.append({
                    "estructura":"Grid File",
                    "puntos_encontrados":len(pts),
                    "nodos_visitados":cells_v,
                    "tiempo_s":t_elapsed
                })
            except Exception as e:
                results_rango.append({"estructura":"Grid File","error":str(e)})

            # R-Tree
            try:
                pts, visited, t_elapsed = rt.buscar_rango(caja)
                results_rango.append({
                    "estructura":"R-Tree",
                    "puntos_encontrados":len(pts),
                    "nodos_visitados":visited,
                    "tiempo_s":t_elapsed
                })
            except Exception as e:
                results_rango.append({"estructura":"R-Tree","error":str(e)})

            # GUARDAR RESULTADOS PERSISTENTES
            st.session_state.bench_rango = {
                "caja": caja,
                "results": results_rango
            }

        # --------------------------------
        # 2) MOSTRAR RESULTADOS SI EXISTEN
        # --------------------------------
        if "bench_rango" in st.session_state:

            b = st.session_state.bench_rango
            df_rg = pd.DataFrame(b["results"])

            st.subheader("Resultados Rango")
            st.dataframe(df_rg)

            # Mapa
            mapa_rr = crear_mapa_base()
            min_lon, max_lon, min_lat, max_lat = b["caja"]
            rect = [
                [min_lat, min_lon],
                [min_lat, max_lon],
                [max_lat, max_lon],
                [max_lat, min_lon],
                [min_lat, min_lon]
            ]
            folium.Polygon(rect, color="#ffa600", weight=2, fill=True, fill_opacity=0.05).add_to(mapa_rr)

            for row in b["results"]:
                if "error" in row:
                    continue
                est = row["estructura"]
                color = colores.get(est, "#444444")

                try:
                    if est == "KD-Tree" and hasattr(kd, "consulta_rango"):
                        pts, _, _ = kd.consulta_rango(b["caja"])
                    elif est == "Quadtree":
                        pts, _, _ = qt.consulta_rango(b["caja"])
                    elif est == "Grid File":
                        pts, _, _ = gf.consulta_rango(b["caja"])
                    elif est == "R-Tree":
                        pts, _, _ = rt.buscar_rango(b["caja"])
                    else:
                        pts = []
                except:
                    pts = []

                for lon, lat, meta in pts:
                    folium.CircleMarker([lat, lon], radius=4, color=color, fill=True).add_to(mapa_rr)

            st_folium(mapa_rr, width=900, height=600)

        fig, ax = plt.subplots(1,2, figsize=(10,4))
        try:
            df_rg_plot = df_rg.dropna(subset=["tiempo_s"]).set_index("estructura")
            df_rg_plot["tiempo_s"].plot.bar(ax=ax[0], title="Tiempo Rango (s)")
        except Exception:
            ax[0].text(0.5,0.5,"No hay datos de tiempo", ha="center")
        try:
            df_rg_plot2 = df_rg.dropna(subset=["nodos_visitados"]).set_index("estructura")
            df_rg_plot2["nodos_visitados"].plot.bar(ax=ax[1], title="Nodos/Celdas visitados")
        except Exception:
            ax[1].text(0.5,0.5,"No hay datos de nodos", ha="center")

        plt.tight_layout()
        st.pyplot(fig)

        # ---- Calcular densidades ----

        num_celdas = gf.rows * gf.cols
        dens_gf = len(puntos) / num_celdas

        total_mbrs = contar_mbrs_rtree(rt)
        dens_rt = len(puntos) / total_mbrs

        fig2, ax = plt.subplots(figsize=(3,2))
        df_dens = pd.DataFrame({
            "estructura": ["Grid File", "R-Tree"],
            "densidad": [dens_gf, dens_rt]
        }).set_index("estructura")

        df_dens["densidad"].plot.bar(ax=ax, title="Factor de densidad (GF vs R-Tree)")
        st.pyplot(fig2, use_container_width=False)


        csv_buf = io.StringIO()
        df_rg.to_csv(csv_buf, index=False)
        st.download_button("Descargar CSV (Rango)", data=csv_buf.getvalue().encode("utf-8"), file_name="bench_rango.csv", mime="text/csv")

    st.write("Fin del módulo de comparación.")

# --------------------------
# FIN
# --------------------------
st.sidebar.markdown("---")
st.sidebar.write("Sugerencia: cambia de estructura en la lista para probar las otras.")

