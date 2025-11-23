"""
Grid File uniforme adaptado al espacio de los datos.
- Grilla rows x cols (por defecto 20x20)
- Distribuye puntos en celdas
- Consulta por rango (rectángulo)
- Vecino más cercano (búsqueda por expansión de anillos de celdas)
- Estadísticas
"""
from typing import List, Tuple, Optional, Any, Dict
import time
import math

PuntoConMeta = Tuple[float, float, Any]   # (lon, lat, meta)
Caja = Tuple[float, float, float, float]  # (min_lon, max_lon, min_lat, max_lat)

def distancia_haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    R = 6371000.0
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))

def caja_min_distancia(caja: Caja, q: Tuple[float, float]) -> float:
    min_lon, max_lon, min_lat, max_lat = caja
    qlon, qlat = q
    clon = max(min_lon, min(max_lon, qlon))
    clat = max(min_lat, min(max_lat, qlat))
    return distancia_haversine_m((clon, clat), q)

class GridFile:
    def __init__(self, limites: Caja, rows: int = 20, cols: int = 20):
        """
        limites: (min_lon, max_lon, min_lat, max_lat)
        rows, cols: tamaño de la grilla
        """
        self.limites = limites
        self.rows = int(rows)
        self.cols = int(cols)
        self.min_lon, self.max_lon, self.min_lat, self.max_lat = limites
        # ancho por celda
        self.cell_w = (self.max_lon - self.min_lon) / self.cols
        self.cell_h = (self.max_lat - self.min_lat) / self.rows
        # celdas: dict (r,c) -> list de puntos
        self.celdas: Dict[Tuple[int,int], List[PuntoConMeta]] = {}
        # inicializar celdas vacías (opcional)
        for r in range(self.rows):
            for c in range(self.cols):
                self.celdas[(r,c)] = []
        self.total_puntos = 0

    def _indice_celda(self, lon: float, lat: float) -> Tuple[int,int]:
        """Devuelve (row, col) de la celda que contiene (lon, lat)."""
        # normalizar y clamp
        if lon <= self.min_lon:
            col = 0
        elif lon >= self.max_lon:
            col = self.cols - 1
        else:
            col = int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.cols)
            col = min(max(col, 0), self.cols - 1)

        if lat <= self.min_lat:
            row = 0
        elif lat >= self.max_lat:
            row = self.rows - 1
        else:
            row = int((lat - self.min_lat) / (self.max_lat - self.min_lat) * self.rows)
            row = min(max(row, 0), self.rows - 1)

        return (row, col)

    def _bbox_celda(self, row: int, col: int) -> Caja:
        """Retorna la caja (min_lon, max_lon, min_lat, max_lat) de la celda."""
        min_lon = self.min_lon + col * self.cell_w
        max_lon = min_lon + self.cell_w
        min_lat = self.min_lat + row * self.cell_h
        max_lat = min_lat + self.cell_h
        return (min_lon, max_lon, min_lat, max_lat)

    def insertar(self, punto: PuntoConMeta) -> bool:
        lon, lat = punto[0], punto[1]
        r, c = self._indice_celda(lon, lat)
        self.celdas[(r,c)].append(punto)
        self.total_puntos += 1
        return True

    def construir(self, lista_puntos: List[PuntoConMeta]):
        """Insertar todos los puntos en la grilla."""
        # limpiar antes
        for k in self.celdas.keys():
            self.celdas[k] = []
        self.total_puntos = 0
        for p in lista_puntos:
            self.insertar(p)

    def consulta_rango(self, caja: Caja):
        """Devuelve puntos en caja, celdas visitadas, tiempo."""
        inicio = time.perf_counter()
        found = []
        visited = 0

        # calcular rango de celdas a chequear
        min_lon, max_lon, min_lat, max_lat = caja

        # indices
        r1, c1 = self._indice_celda(min_lon, min_lat)
        r2, c2 = self._indice_celda(max_lon, max_lat)

        # asegurar orden
        rmin, rmax = min(r1, r2), max(r1, r2)
        cmin, cmax = min(c1, c2), max(c1, c2)

        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                visited += 1
                for p in self.celdas[(r,c)]:
                    lon, lat = p[0], p[1]
                    if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat):
                        found.append(p)

        dur = time.perf_counter() - inicio
        return found, visited, dur

    def vecino_mas_cercano(self, consulta: Tuple[float, float]):
        """
        Vecino más cercano basado en expansión por celdas (anillos).
        Retorna (punto, distancia_m, celdas_visitadas, tiempo, recorrido)
        donde RECORRIDO es la lista ORDENADA de celdas visitadas.
        """
        inicio = time.perf_counter()
        best = {"p": None, "d": float("inf")}
        visited = 0
        recorrido = []       
        visited_set = set()   # para evitar repetir celdas

        # celda inicial donde cae el punto
        row0, col0 = self._indice_celda(consulta[0], consulta[1])

        # función auxiliar para podar
        def min_dist_a_celda(r, c):
            bbox = self._bbox_celda(r, c)
            return caja_min_distancia(bbox, consulta)

        # ---------- 1. Revisar la celda central ----------
        if 0 <= row0 < self.rows and 0 <= col0 < self.cols:
            visited += 1
            visited_set.add((row0, col0))
            recorrido.append((row0, col0))

            for p in self.celdas[(row0, col0)]:
                d = distancia_haversine_m((p[0], p[1]), consulta)
                if d < best["d"]:
                    best["p"] = p
                    best["d"] = d

        # ---------- 2. Expandir anillos ----------
        max_ring = max(self.rows, self.cols)

        for ring in range(1, max_ring):

            rmin = max(0, row0 - ring)
            rmax = min(self.rows - 1, row0 + ring)
            cmin = max(0, col0 - ring)
            cmax = min(self.cols - 1, col0 + ring)

            cells_to_check = []

            # parte superior e inferior
            for c in range(cmin, cmax + 1):
                cells_to_check.append((rmin, c))
                cells_to_check.append((rmax, c))

            # lados
            for r in range(rmin + 1, rmax):
                cells_to_check.append((r, cmin))
                cells_to_check.append((r, cmax))

            # limpiar duplicados
            cells_to_check = [c for c in dict.fromkeys(cells_to_check)
                            if 0 <= c[0] < self.rows and 0 <= c[1] < self.cols]

            # ordenar por distancia mínima posible
            cells_to_check.sort(key=lambda rc: min_dist_a_celda(rc[0], rc[1]))

            # poda global
            if cells_to_check:
                r1, c1 = cells_to_check[0]
                if min_dist_a_celda(r1, c1) > best["d"]:
                    break

            # explorar celdas del anillo
            for (r, c) in cells_to_check:

                if (r, c) in visited_set:
                    continue

                visited_set.add((r, c))
                visited += 1
                recorrido.append((r, c)) 

                # poda por celda individual
                if min_dist_a_celda(r, c) > best["d"]:
                    continue

                # revisar puntos de la celda
                for p in self.celdas[(r, c)]:
                    d = distancia_haversine_m((p[0], p[1]), consulta)
                    if d < best["d"]:
                        best["p"] = p
                        best["d"] = d

        dur = time.perf_counter() - inicio

        if best["p"] is None:
            return None, None, visited, dur, recorrido

        return best["p"], best["d"], visited, dur, recorrido


    def estadisticas(self):
        total_nonempty = sum(1 for v in self.celdas.values() if v)
        avg_por_celda = (self.total_puntos / total_nonempty) if total_nonempty > 0 else 0
        return {
            "rows": self.rows,
            "cols": self.cols,
            "celdas_totales": self.rows * self.cols,
            "celdas_usadas": total_nonempty,
            "puntos_totales": self.total_puntos,
            "promedio_por_celda_usada": avg_por_celda
        }

    def factor_carga(self):
        celdas_ocupadas = sum(1 for k in self.celdas if len(self.celdas[k]) > 0)
        total_puntos = sum(len(self.celdas[k]) for k in self.celdas)
        if celdas_ocupadas == 0:
            return 0
        return total_puntos / celdas_ocupadas