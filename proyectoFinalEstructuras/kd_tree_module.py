#KD-Tree clásico (2D) preparado para coordenadas reales (lat, lon CSV input).
from typing import List, Tuple, Optional, Any
import math
import time
import sys

# -------------------------------
# Alias de tipos
# -------------------------------
Punto = Tuple[float, float]                   # (lon, lat)
PuntoConMeta = Tuple[float, float, Any]       # (lon, lat, meta)


# ============================================================
# NODO KD
# ============================================================
class NodoKD:
    __slots__ = ("punto", "izq", "der", "eje", "lineas_visita")

    def __init__(self, punto: PuntoConMeta, eje: int):
        self.punto: PuntoConMeta = punto
        self.izq: Optional["NodoKD"] = None
        self.der: Optional["NodoKD"] = None
        self.eje: int = eje   # 0 = lon, 1 = lat
        self.lineas_visita = []   # para visualización


# ============================================================
# KD-TREE
# ============================================================
class KDTree:
    def __init__(self):
        self.raiz: Optional[NodoKD] = None
        self.tamano: int = 0

    # ----------------------------------------------------------
    # Construcción del árbol
    # ----------------------------------------------------------
    def construir(self, puntos: List[PuntoConMeta]) -> None:
        """Construye el KDTree desde una lista de (lon, lat, meta)."""

        def construir_rec(lista: List[PuntoConMeta], profundidad: int) -> Optional[NodoKD]:
            if not lista:
                return None

            eje = profundidad % 2
            lista.sort(key=lambda p: p[eje])
            mid = len(lista) // 2

            nodo = NodoKD(lista[mid], eje)
            nodo.izq = construir_rec(lista[:mid], profundidad + 1)
            nodo.der = construir_rec(lista[mid + 1:], profundidad + 1)
            return nodo

        self.raiz = construir_rec(puntos.copy(), 0)
        self.tamano = len(puntos)

    # ----------------------------------------------------------
    # Distancia Haversine
    # ----------------------------------------------------------
    @staticmethod
    def _haversine_m(a: Punto, b: Punto) -> float:
        lon1, lat1 = a
        lon2, lat2 = b
        lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        R = 6371000.0
        hav = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2 * R * math.asin(min(1.0, math.sqrt(hav)))

    # ----------------------------------------------------------
    # Vecino más cercano
    # ----------------------------------------------------------
    def vecino_mas_cercano(self, consulta: Punto):
        """
        Retorna:
          (mejor_punto, distancia_m, nodos_visitados, tiempo, recorrido)
        """
        if self.raiz is None:
            return None, None, 0, 0.0, []

        inicio = time.perf_counter()
        nodos_vis = 0
        mejor = {"nodo": None, "dist": float("inf")}
        recorrido = []   # <-- para visualizar

        def buscar(nodo: Optional[NodoKD]):
            nonlocal nodos_vis, mejor

            if nodo is None:
                return

            nodos_vis += 1
            recorrido.append(nodo.punto)  # guardamos visita

            punto_actual = (nodo.punto[0], nodo.punto[1])
            d = self._haversine_m(consulta, punto_actual)

            if d < mejor["dist"]:
                mejor["dist"] = d
                mejor["nodo"] = nodo

            eje = nodo.eje
            qcoord = consulta[eje]
            ncoord = nodo.punto[eje]

            # Elegir rama principal
            primero, segundo = (
                (nodo.izq, nodo.der) if qcoord < ncoord else (nodo.der, nodo.izq)
            )

            if primero:
                buscar(primero)

            # Decidir si visitar la otra rama
            if eje == 0:
                punto_plano = (ncoord, consulta[1])
            else:
                punto_plano = (consulta[0], ncoord)

            dist_plano = self._haversine_m(consulta, punto_plano)

            if dist_plano < mejor["dist"]:
                if segundo:
                    buscar(segundo)

        buscar(self.raiz)
        duracion = time.perf_counter() - inicio

        if mejor["nodo"] is None:
            return None, None, nodos_vis, duracion, recorrido

        punto_final = mejor["nodo"].punto
        return punto_final, mejor["dist"], nodos_vis, duracion, recorrido

    # ----------------------------------------------------------
    # Consulta por rango
    # ----------------------------------------------------------
    def consulta_rango(self, caja: Tuple[float, float, float, float]):
        """
        Caja: (min_lon, max_lon, min_lat, max_lat)
        Retorna: (puntos_en_rango, nodos_visitados, tiempo)
        """
        if self.raiz is None:
            return [], 0, 0.0, []

        inicio = time.perf_counter()
        nodos_vis = 0
        salida: List[PuntoConMeta] = []

        def dentro(pt: PuntoConMeta):
            lon, lat = pt[0], pt[1]
            return (
                caja[0] <= lon <= caja[1] and
                caja[2] <= lat <= caja[3]
            )

        def rec(nodo: Optional[NodoKD]):
            nonlocal nodos_vis
            if nodo is None:
                return

            nodos_vis += 1

            lon = nodo.punto[0]
            lat = nodo.punto[1]
            eje = nodo.eje

            if dentro(nodo.punto):
                salida.append(nodo.punto)

            if eje == 0:   # división por lon
                if nodo.izq and caja[0] <= lon:
                    rec(nodo.izq)
                if nodo.der and caja[1] >= lon:
                    rec(nodo.der)
            else:          # división por lat
                if nodo.izq and caja[2] <= lat:
                    rec(nodo.izq)
                if nodo.der and caja[3] >= lat:
                    rec(nodo.der)

        rec(self.raiz)
        duracion = time.perf_counter() - inicio
        return salida, nodos_vis, duracion

    # ----------------------------------------------------------
    # Todos los puntos
    # ----------------------------------------------------------
    def todos_los_puntos(self) -> List[PuntoConMeta]:
        out = []

        def rec(nodo: Optional[NodoKD]):
            if nodo:
                out.append(nodo.punto)
                rec(nodo.izq)
                rec(nodo.der)

        rec(self.raiz)
        return out
