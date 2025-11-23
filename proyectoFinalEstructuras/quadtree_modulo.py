"""
Quadtree de puntos para coordenadas (longitud, latitud).
Se usa para índices espaciales y búsqueda eficiente.
"""

from typing import List, Tuple, Optional, Any
import math
import time

PuntoConMeta = Tuple[float, float, Any]   # (lon, lat, metadata)
Caja = Tuple[float, float, float, float]  # (min_lon, max_lon, min_lat, max_lat)

# -------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------
def distancia_haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calcula la distancia en metros entre dos coordenadas usando Haversine."""
    lon1, lat1 = a
    lon2, lat2 = b
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    R = 6371000.0  # radio de la Tierra en metros
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))

def caja_contiene(caja: Caja, punto: PuntoConMeta) -> bool:
    """Verifica si una caja contiene un punto."""
    min_lon, max_lon, min_lat, max_lat = caja
    lon, lat = punto[0], punto[1]
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

def cajas_intersectan(a: Caja, b: Caja) -> bool:
    """Verifica si dos cajas se solapan."""
    amin_lon, amax_lon, amin_lat, amax_lat = a
    bmin_lon, bmax_lon, bmin_lat, bmax_lat = b
    return not (amax_lon < bmin_lon or amin_lon > bmax_lon or amax_lat < bmin_lat or amin_lat > bmax_lat)

# -------------------------------------------------------------
# NODO DEL QUADTREE
# -------------------------------------------------------------
class NodoQuadtree:
    """Nodo del Quadtree con caja, puntos y subdivisiones."""
    __slots__ = ("caja", "puntos", "hijos", "dividido", "profundidad", "centro_lon",
        "centro_lat")

    def __init__(self, caja: Caja, profundidad: int = 0):
        self.caja = caja
        self.puntos: List[PuntoConMeta] = []
        self.hijos: Optional[List["NodoQuadtree"]] = None
        self.dividido = False
        self.profundidad = profundidad
        min_lon, max_lon, min_lat, max_lat = caja
        self.centro_lon = (min_lon + max_lon) / 2
        self.centro_lat = (min_lat + max_lat) / 2


    def subdividir(self):
        """Divide el nodo en 4 cuadrantes."""
        min_lon, max_lon, min_lat, max_lat = self.caja
        mitad_lon = (min_lon + max_lon) / 2
        mitad_lat = (min_lat + max_lat) / 2

        # Cuadrantes: arriba-izquierda, arriba-derecha, abajo-izquierda, abajo-derecha
        ul = (min_lon, mitad_lon, mitad_lat, max_lat)
        ur = (mitad_lon, max_lon, mitad_lat, max_lat)
        ll = (min_lon, mitad_lon, min_lat, mitad_lat)
        lr = (mitad_lon, max_lon, min_lat, mitad_lat)

        self.hijos = [
            NodoQuadtree(ul, self.profundidad + 1),
            NodoQuadtree(ur, self.profundidad + 1),
            NodoQuadtree(ll, self.profundidad + 1),
            NodoQuadtree(lr, self.profundidad + 1)
        ]
        self.dividido = True

# -------------------------------------------------------------
# QUADTREE PRINCIPAL
# -------------------------------------------------------------
class Quadtree:
    """Estructura Quadtree completa para indexar puntos geográficos."""

    def __init__(self, limites: Caja, capacidad: int = 8, profundidad_max: int = 20):
        self.raiz = NodoQuadtree(limites)
        self.capacidad = capacidad
        self.profundidad_max = profundidad_max
        self.total_puntos = 0
        self._last_visit_rects = []

    # ---------------------------------------------------------
    # Inserción
    # ---------------------------------------------------------
    def insertar(self, punto: PuntoConMeta) -> bool:
        """Inserta un punto dentro del Quadtree."""
        if not caja_contiene(self.raiz.caja, punto):
            return False
        ok = self._insertar_rec(self.raiz, punto)
        if ok:
            self.total_puntos += 1
        return ok

    def _insertar_rec(self, nodo: NodoQuadtree, punto: PuntoConMeta) -> bool:
        """Inserción recursiva en el nodo adecuado."""
        if not caja_contiene(nodo.caja, punto):
            return False

        # Si el nodo no está dividido aún
        if not nodo.dividido:
            # Si aún puede almacenar puntos
            if len(nodo.puntos) < self.capacidad or nodo.profundidad >= self.profundidad_max:
                nodo.puntos.append(punto)
                return True
            # Si se llena, se subdivide
            nodo.subdividir()
            # Redistribuye puntos anteriores
            puntos_previos = nodo.puntos
            nodo.puntos = []
            for p in puntos_previos:
                for h in nodo.hijos:
                    if caja_contiene(h.caja, p):
                        h.puntos.append(p)
                        break

        # Insertar en uno de los hijos
        for h in nodo.hijos:
            if caja_contiene(h.caja, punto):
                return self._insertar_rec(h, punto)

        nodo.puntos.append(punto)  # caso extremo
        return True

    # ---------------------------------------------------------
    # Construcción desde lista
    # ---------------------------------------------------------
    def construir(self, lista_puntos: List[PuntoConMeta]):
        """Construye el Quadtree insertando cada punto."""
        self.total_puntos = 0
        for p in lista_puntos:
            self.insertar(p)

    # ---------------------------------------------------------
    # Consulta por rango
    # ---------------------------------------------------------
    def consulta_rango(self, caja: Caja):
        """Retorna puntos dentro de una caja, nodos visitados y tiempo."""
        inicio = time.perf_counter()
        encontrados: List[PuntoConMeta] = []
        nodos_visitados = 0

        def rec(nodo: NodoQuadtree):
            nonlocal nodos_visitados
            nodos_visitados += 1

            if not cajas_intersectan(nodo.caja, caja):
                return

            for p in nodo.puntos:
                if caja_contiene(caja, p):
                    encontrados.append(p)

            if nodo.dividido:
                for h in nodo.hijos:
                    rec(h)
        
        rec(self.raiz)
        duracion = time.perf_counter() - inicio
        return encontrados, nodos_visitados, duracion

    # ---------------------------------------------------------
    # Vecino más cercano
    # ---------------------------------------------------------
    def vecino_mas_cercano(self, consulta: Tuple[float, float]):
        """
        Retorna:
        (mejor_punto, distancia_m, nodos_visitados, tiempo_s, recorrido)
        recorrido = lista de puntos visitados para visualización
        """

        inicio = time.perf_counter()
        mejor = {"punto": None, "dist": float("inf")}
        nodos_visitados = 0
        recorrido = []  # ← para dibujar el recorrido en el mapa

        def distancia_min_caja(caja: Caja, q):
            """Distancia mínima posible entre la caja del nodo y el punto de consulta."""
            min_lon, max_lon, min_lat, max_lat = caja
            qlon, qlat = q

            # proyectar el punto sobre la caja
            clon = max(min_lon, min(max_lon, qlon))
            clat = max(min_lat, min(max_lat, qlat))

            return distancia_haversine_m((clon, clat), q)

        def rec(nodo: NodoQuadtree):
            nonlocal nodos_visitados, mejor

            nodos_visitados += 1
            # registrar este nodo (lo dibujaremos en el mapa)
            recorrido.append((nodo.centro_lon, nodo.centro_lat))

            # poda
            if distancia_min_caja(nodo.caja, consulta) > mejor["dist"]:
                return

            # revisar puntos del nodo
            for p in nodo.puntos:
                d = distancia_haversine_m((p[0], p[1]), consulta)
                if d < mejor["dist"]:
                    mejor["dist"] = d
                    mejor["punto"] = p

            # revisar hijos en orden optimizado
            if nodo.dividido:
                hijos_ordenados = sorted(
                    nodo.hijos,
                    key=lambda h: distancia_min_caja(h.caja, consulta)
                )
                for h in hijos_ordenados:
                    rec(h)

        rec(self.raiz)
        duracion = time.perf_counter() - inicio

        return mejor["punto"], mejor["dist"], nodos_visitados, duracion, recorrido

    # ---------------------------------------------------------
    # Estadísticas
    # ---------------------------------------------------------
    def _recolectar_nodos(self):
        lista = []
        def rec(n):
            lista.append(n)
            if n.dividido:
                for h in n.hijos:
                    rec(h)
        rec(self.raiz)
        return lista

    def estadisticas(self):
        """Retorna dict con nodos totales, hojas, profundidad máxima y puntos."""
        nodos = self._recolectar_nodos()
        hojas = sum(1 for n in nodos if not n.dividido)
        prof_max = max((n.profundidad for n in nodos), default=0)
        return {
            "nodos_totales": len(nodos),
            "hojas": hojas,
            "profundidad_max": prof_max,
            "puntos_totales": self.total_puntos
        }
