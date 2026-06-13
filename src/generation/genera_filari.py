
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.ops import split


def seziona_con_linee_parallele(polygon: Polygon, larghezza: float, orientamento_xy: tuple = None):
    """
    Divide un poligono in strisce parallele secondo un vettore di orientamento specificato,
    oppure automaticamente lungo il lato più lungo del perimetro.

    Args:
        polygon (Polygon): La sottocella da sezionare.
        larghezza (float): La distanza tra le linee parallele.
        orientamento_xy (tuple): Direzione desiderata (es. (1, 0)). Se None, usa il lato più lungo.

    Returns:
        List[Polygon]: Lista di sottosezioni ottenute.
    """
    if polygon.is_empty or not polygon.is_valid:
        return []

    if orientamento_xy:
        dx, dy = orientamento_xy
    else:
        # Calcolo della direzione del lato più lungo
        coords = list(polygon.exterior.coords)
        max_len = 0
        dx = dy = 0
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            length = np.hypot(x1 - x0, y1 - y0)
            if length > max_len:
                max_len = length
                dx = x1 - x0
                dy = y1 - y0

    # Normalizzazione del vettore direzione
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm

    # Direzione ortogonale alle linee (verso la quale si scansiona)
    ox, oy = -dy, dx

    # Proiezione dei vertici per determinare l'estensione nella direzione ortogonale
    coords = np.array(polygon.exterior.coords)
    projections = coords @ np.array([ox, oy])
    min_proj, max_proj = projections.min(), projections.max()

    num_lines = int(np.ceil((max_proj - min_proj) / larghezza)) + 1
    lines = []
    for i in range(num_lines):
        offset = min_proj + i * larghezza
        cx, cy = ox * offset, oy * offset
        line = LineString([
            (cx - dx * 10000, cy - dy * 10000),
            (cx + dx * 10000, cy + dy * 10000)
        ])
        lines.append(line)

    result = [polygon]
    for line in lines:
        new_result = []
        for poly in result:
            if poly.is_empty or not poly.is_valid:
                continue
            split_polys = split(poly, line)
            new_result.extend(split_polys.geoms if hasattr(split_polys, 'geoms') else [split_polys])
        result = new_result

    return [p for p in result if isinstance(p, Polygon) and not p.is_empty]


def plot_polygons(polygons, title="Suddivisione", cmap_name='viridis'):
    fig, ax = plt.subplots()
    cmap = plt.colormaps.get_cmap(cmap_name)
    for i, poly in enumerate(polygons):
        color = cmap(i / max(len(polygons) - 1, 1))
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor=color, edgecolor='black', alpha=0.6)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Esecuzione esempio di sezionamento parallelo...")

    # Esempio 1: Rettangolo
    rettangolo = Polygon([(0, 0), (6, 0), (6, 2), (0, 2)])
    sezioni1 = seziona_con_linee_parallele(rettangolo, larghezza=0.5)
    plot_polygons(sezioni1, title="Rettangolo - Orientamento automatico")

    # Esempio 2: Trapezio rettangolo
    trapezio = Polygon([(0, 0), (4, 0), (3, 2), (0, 2)])
    sezioni2 = seziona_con_linee_parallele(trapezio, larghezza=0.5)
    plot_polygons(sezioni2, title="Trapezio - Orientamento automatico")

    # Esempio 3: Parallelogramma con orientamento manuale
    parallelogramma = Polygon([(0, 0), (4, 0), (5, 2), (1, 2)])
    sezioni3 = seziona_con_linee_parallele(parallelogramma, larghezza=0.5, orientamento_xy=(2, 1))
    plot_polygons(sezioni3, title="Parallelogramma - Orientamento manuale (2,1)")

    print("Sezionamento completato.")
