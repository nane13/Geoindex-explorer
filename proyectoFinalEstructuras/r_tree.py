"""
Implementación simple de R-Tree (Guttman) con split LINEAL (clásico).
Soporta:
- Inserción de puntos (lon, lat, meta)
- Búsqueda por rango (rectángulo)
- Vecino más cercano (Búsqueda por prioridad de MBR)
- Visualización (devuelve MBRs)
"""

from typing import Any, List, Tuple, Optional
import math
import time
import heapq
import itertools

PuntoConMeta = Tuple[float, float, Any]  # (lon, lat, meta)
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

def caja_area(caja: Caja) -> float:
    # área en grados (aprox), usada solo para comparación de MBRs
    min_lon, max_lon, min_lat, max_lat = caja
    return (max_lon - min_lon) * (max_lat - min_lat)

def caja_union(a: Caja, b: Caja) -> Caja:
    min_lon = min(a[0], b[0])
    max_lon = max(a[1], b[1])
    min_lat = min(a[2], b[2])
    max_lat = max(a[3], b[3])
    return (min_lon, max_lon, min_lat, max_lat)

def caja_intersecta(a: Caja, b: Caja) -> bool:
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

def caja_distancia_minima(caja: Caja, punto: Tuple[float, float]) -> float:
    qlon, qlat = punto
    min_lon, max_lon, min_lat, max_lat = caja
    clon = max(min_lon, min(max_lon, qlon))
    clat = max(min_lat, min(max_lat, qlat))
    return distancia_haversine_m((clon, clat), (qlon, qlat))

class RTreeEntry:
    def __init__(self, caja: Caja, hijo=None, punto: PuntoConMeta=None):
        self.caja = caja
        self.hijo = hijo  # None si es entrada de hoja; si no, referencia a RTreeNode
        self.punto = punto  # solo para hojas (punto es (lon, lat, meta))

class RTreeNode:
    def __init__(self, hoja: bool, max_entries: int = 8):
        self.hoja = hoja
        self.entries: List[RTreeEntry] = []
        self.max_entries = max_entries
        self.parent = None

    def es_lleno(self) -> bool:
        return len(self.entries) > self.max_entries

    def caja_del_nodo(self) -> Optional[Caja]:
        if not self.entries:
            return None
        caja = self.entries[0].caja
        for e in self.entries[1:]:
            caja = caja_union(caja, e.caja)
        return caja

class RTree:
    def __init__(self, max_entries: int = 8, min_entries: Optional[int] = None):
        self.max_entries = max_entries
        self.min_entries = min_entries or (max_entries // 2)
        self.root = RTreeNode(hoja=True, max_entries=max_entries)
        self._n_puntos = 0

    # ---------- Inserción ----------
    def _caja_de_punto(self, p: PuntoConMeta) -> Caja:
        lon, lat, _ = p
        eps = 1e-9
        return (lon - eps, lon + eps, lat - eps, lat + eps)

    def insertar(self, p: PuntoConMeta):
        """Inserta un punto en el árbol."""
        leaf = self._elegir_hoja(self.root, p)
        entry = RTreeEntry(self._caja_de_punto(p), hijo=None, punto=p)
        leaf.entries.append(entry)
        self._n_puntos += 1
        self._ajustar_arbol(leaf)

    def _elegir_hoja(self, node: RTreeNode, p: PuntoConMeta) -> RTreeNode:
        if node.hoja:
            return node
        # escoger entrada que requiera menor incremento de área
        lon, lat = p[0], p[1]
        best = None
        best_inc = None
        for e in node.entries:
            orig_area = caja_area(e.caja)
            new_box = caja_union(e.caja, self._caja_de_punto(p))
            inc = caja_area(new_box) - orig_area
            if best is None or inc < best_inc or (inc == best_inc and orig_area < caja_area(best.caja)):
                best = e
                best_inc = inc
        return self._elegir_hoja(best.hijo, p)

    def _ajustar_arbol(self, node: RTreeNode):
        """Ajusta hacia arriba: si hay overflow, split; si root overflow, crear nuevo root."""
        current = node
        while current is not None:
            if len(current.entries) > current.max_entries:
                n1, n2 = self._split_node(current)
                if current == self.root:
                    # crear nuevo root
                    new_root = RTreeNode(hoja=False, max_entries=self.max_entries)
                    e1 = RTreeEntry(n1.caja_del_nodo(), hijo=n1)
                    e2 = RTreeEntry(n2.caja_del_nodo(), hijo=n2)
                    n1.parent = new_root
                    n2.parent = new_root
                    new_root.entries = [e1, e2]
                    self.root = new_root
                    return
                else:
                    parent = current.parent
                    found_idx = None
                    for i, ent in enumerate(parent.entries):
                        if ent.hijo == current:
                            found_idx = i
                            break
                    if found_idx is not None:
                        parent.entries.pop(found_idx)
                    # crear entradas para n1 y n2
                    e1 = RTreeEntry(n1.caja_del_nodo(), hijo=n1)
                    e2 = RTreeEntry(n2.caja_del_nodo(), hijo=n2)
                    n1.parent = parent
                    n2.parent = parent
                    parent.entries.append(e1)
                    parent.entries.append(e2)
                    current = parent
                    continue
            else:
                # actualizar la caja de la entrada en el padre
                if current.parent is not None:
                    for ent in current.parent.entries:
                        if ent.hijo == current:
                            ent.caja = current.caja_del_nodo()
                            break
            current = current.parent

    # ---------- Split LINEAL (Guttman) ----------
    def _split_node(self, node: RTreeNode) -> Tuple[RTreeNode, RTreeNode]:
        """
        Implementación del split LINEAL: elegir dos seeds por mayor separación en una dimensión.
        Retorna dos nodos nuevos con repartición.
        """
        entries = node.entries.copy()

        # 1) elegir par de seeds
        # Para cada dimensión, encontrar el rectángulo con min y max low/high y calcular normalized separation.
        def pick_seeds(entries):
            # dimensiones: lon (0 low,1 high), lat (2 low,3 high)
            # para lon: consideramos low = min_lon, high = max_lon
            min_low_lon = min(e.caja[0] for e in entries)
            max_high_lon = max(e.caja[1] for e in entries)
            min_low_lat = min(e.caja[2] for e in entries)
            max_high_lat = max(e.caja[3] for e in entries)

            low_lon_e = min(entries, key=lambda e: e.caja[0])
            high_lon_e = max(entries, key=lambda e: e.caja[1])
            low_lat_e = min(entries, key=lambda e: e.caja[2])
            high_lat_e = max(entries, key=lambda e: e.caja[3])

            sep_lon = abs(high_lon_e.caja[1] - low_lon_e.caja[0])
            sep_lat = abs(high_lat_e.caja[3] - low_lat_e.caja[2])

            if sep_lon >= sep_lat:
                return low_lon_e, high_lon_e
            else:
                return low_lat_e, high_lat_e

        seed1, seed2 = pick_seeds(entries)
        group1 = RTreeNode(hoja=node.hoja, max_entries=node.max_entries)
        group2 = RTreeNode(hoja=node.hoja, max_entries=node.max_entries)
        group1.parent = node.parent
        group2.parent = node.parent

        group1.entries.append(seed1)
        group2.entries.append(seed2)

        remaining = [e for e in entries if e is not seed1 and e is not seed2]

        # 2) distribuir entries restantes por criterio de incremento de área
        while remaining:
            # si queda tan pocos que hay que asignarlos para llenar mínimos, asignar y salir
            if (len(group1.entries) + len(remaining)) == self.min_entries:
                group1.entries.extend(remaining)
                break
            if (len(group2.entries) + len(remaining)) == self.min_entries:
                group2.entries.extend(remaining)
                break

            # escoger entry que produce la mayor preferencia para uno de los grupos
            best = None
            best_diff = None
            for e in remaining:
                area1 = caja_area(caja_union(group1.caja_del_nodo() or e.caja, e.caja))
                area2 = caja_area(caja_union(group2.caja_del_nodo() or e.caja, e.caja))
                inc1 = area1 - (caja_area(group1.caja_del_nodo()) if group1.caja_del_nodo() else 0)
                inc2 = area2 - (caja_area(group2.caja_del_nodo()) if group2.caja_del_nodo() else 0)
                diff = abs(inc1 - inc2)
                if best is None or diff > best_diff:
                    best = e
                    best_diff = diff
                    best_inc1 = inc1
                    best_inc2 = inc2
            # asignar a grupo con menor incremento; en empate, al grupo con menor area actual
            if best_inc1 < best_inc2:
                group1.entries.append(best)
            elif best_inc2 < best_inc1:
                group2.entries.append(best)
            else:
                a1 = caja_area(group1.caja_del_nodo()) if group1.caja_del_nodo() else 0
                a2 = caja_area(group2.caja_del_nodo()) if group2.caja_del_nodo() else 0
                if a1 <= a2:
                    group1.entries.append(best)
                else:
                    group2.entries.append(best)
            remaining.remove(best)

        # asegurar que cada nodo tenga parent actualizado para hijos
        for g in (group1, group2):
            for ent in g.entries:
                if ent.hijo is not None:
                    ent.hijo.parent = g

        return group1, group2

    # ---------- Búsqueda por rango ----------
    def buscar_rango(self, caja: Caja) -> Tuple[List[PuntoConMeta], int, float]:
        inicio = time.perf_counter()
        found = []
        visited = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            visited += 1
            for e in node.entries:
                if caja_intersecta(e.caja, caja):
                    if node.hoja:
                        # e.punto es el punto real
                        if e.punto is not None:
                            lon, lat, _ = e.punto
                            if caja[0] <= lon <= caja[1] and caja[2] <= lat <= caja[3]:
                                found.append(e.punto)
                    else:
                        stack.append(e.hijo)
        dur = time.perf_counter() - inicio
        return found, visited, dur

    # ---------- Vecino más cercano (priority search) ----------
    def vecino_mas_cercano(self, consulta: Tuple[float, float]):
        """
        Búsqueda del vecino más cercano usando búsqueda best-first.
        Retorna:
            (punto_encontrado, distancia_m, nodos_visitados, tiempo_s, recorrido_mbrs)
        donde recorrido_mbrs es la lista de MBRs de nodos visitados, para visualización.
        """
        inicio = time.perf_counter()
        visited = 0
        best_p = None
        best_d = float("inf")

        # contador anti-tie
        counter = itertools.count()

        # priority queue: (dist_min_posible, tie, item, is_node)
        pq = []

        root_caja = self.root.caja_del_nodo()
        if root_caja is None:
            return None, None, 0, 0.0, []

        recorrido_mbrs = []      

        # encolar raíz
        heapq.heappush(pq, (caja_distancia_minima(root_caja, consulta),
                            next(counter), self.root, True))

        while pq:
            dist_min, _, item, is_node = heapq.heappop(pq)

            # poda global
            if dist_min >= best_d:
                break

            if is_node:
                node: RTreeNode = item
                visited += 1

                # registrar MBR visitado
                node_box = node.caja_del_nodo()
                if node_box:
                    recorrido_mbrs.append(node_box)

                for e in node.entries:
                    if node.hoja:
                        if e.punto is not None:
                            d = distancia_haversine_m((e.punto[0], e.punto[1]), consulta)
                            if d < best_d:
                                best_d = d
                                best_p = e.punto
                    else:
                        dmin = caja_distancia_minima(e.caja, consulta)
                        if dmin < best_d:
                            heapq.heappush(pq, (dmin, next(counter), e.hijo, True))

        dur = time.perf_counter() - inicio

        if best_p is None:
            return None, None, visited, dur, recorrido_mbrs

        return best_p, best_d, visited, dur, recorrido_mbrs

    def factor_carga(self):
        hojas = []
        def recolectar(nodo):
            if nodo.hoja:
                hojas.append(len(nodo.entries))
            else:
                for e in nodo.entries:
                    recolectar(e.hijo)

        recolectar(self.root)

        if not hojas:
            return 0

        return sum(hojas) / len(hojas)

    # ---------- Utilidades ----------
    @property
    def n_puntos(self):
        return self._n_puntos

    def mbrs_a_dibujar(self) -> List[Caja]:
        """Devuelve lista de MBRs (rectángulos) de todo el árbol para visualización."""
        rects = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            c = node.caja_del_nodo()
            if c:
                rects.append(c)
            if not node.hoja:
                for e in node.entries:
                    stack.append(e.hijo)
        return rects
