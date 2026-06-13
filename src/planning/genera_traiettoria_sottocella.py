import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.ops import split # Not directly used in new turn logic, but kept for context

# --- Helper for Bezier curve generation ---
def generate_bezier_curve(p0, p1, p2, p3, num_points=50):
    """Generates points for a cubic Bezier curve."""
    t = np.linspace(0, 1, num_points)[:, np.newaxis] # Add new axis for broadcasting
    curve_points = (
        (1-t)**3 * p0 +
        3 * (1-t)**2 * t * p1 +
        3 * (1-t) * t**2 * p2 +
        t**3 * p3
    )
    return LineString(curve_points)

# --- Original function, slightly modified for better output consistency ---
def genera_traiettorie_parallele(polygon: Polygon, larghezza: float, orientamento_xy: tuple = None):
    """
    Genera le traiettorie centrali (linee guida) di strisce parallele all'interno di un poligono.
    L'orientamento delle strisce può essere specificato o calcolato automaticamente
    lungo il lato più lungo del perimetro del poligono.

    Args:
        polygon (Polygon): Il poligono di input da sezionare.
        larghezza (float): La distanza tra le traiettorie, che corrisponde alla larghezza nominale di ogni striscia.
        orientamento_xy (tuple): Direzione desiderata delle strisce come vettore (es. (1, 0) per orizzontale).
                                 Se None, la direzione viene determinata automaticamente dal lato più lungo del poligono.

    Returns:
        tuple: (List[LineString], tuple, tuple)
               - Lista di oggetti LineString, ognuno rappresentante una traiettoria centrale.
               - Vettore di direzione (dx, dy) lungo le file.
               - Vettore ortogonale (ox, oy) attraverso le file.
    """
    if polygon.is_empty or not polygon.is_valid:
        print("Il poligono è vuoto o non valido.")
        return [], (0,0), (0,0)

    if orientamento_xy:
        dx, dy = orientamento_xy
    else:
        coords = list(polygon.exterior.coords)
        max_len = 0
        dx = dy = 0
        if len(coords) < 2:
            return [], (0,0), (0,0)
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            length = np.hypot(x1 - x0, y1 - y0)
            if length > max_len:
                max_len = length
                dx = x1 - x0
                dy = y1 - y0
        if max_len == 0:
            print("Impossibile determinare un lato più lungo per il poligono.")
            return [], (0,0), (0,0)

    norm = np.hypot(dx, dy)
    if norm == 0:
        print("Il vettore di orientamento è nullo.")
        return [], (0,0), (0,0)
    dx, dy = dx / norm, dy / norm

    ox, oy = -dy, dx # Orthogonal vector

    coords_array = np.array(polygon.exterior.coords)
    projections = coords_array @ np.array([ox, oy])
    min_proj, max_proj = projections.min(), projections.max()

    minx, miny, maxx, maxy = polygon.bounds
    line_length_buffer = np.hypot(maxx - minx, maxy - miny) * 2 # Buffer for long lines

    trajectories = []
    # Start the first trajectory at half width from the minimum projected edge
    current_offset = min_proj + larghezza / 2

    while current_offset < max_proj:
        cx, cy = ox * current_offset, oy * current_offset
        long_line = LineString([
            (cx - dx * line_length_buffer, cy - dy * line_length_buffer),
            (cx + dx * line_length_buffer, cy + dy * line_length_buffer)
        ])
        
        intersection = polygon.intersection(long_line)
        
        if isinstance(intersection, LineString) and not intersection.is_empty:
            trajectories.append(intersection)
        elif hasattr(intersection, 'geoms'):
            for geom in intersection.geoms:
                if isinstance(geom, LineString) and not geom.is_empty:
                    trajectories.append(geom)
        
        current_offset += larghezza

    # Sort trajectories based on their position along the orthogonal vector for consistent ordering
    trajectories.sort(key=lambda line: np.dot(np.array(line.coords[0]), np.array([ox, oy])))

    return trajectories, (dx, dy), (ox, oy)


def get_travel_endpoints(line_string: LineString, row_index: int, row_direction_vector: tuple):
    """
    Determines the start and end points of a LineString based on the assumed travel direction.
    Assumes alternating travel direction for rows.

    Args:
        line_string (LineString): The trajectory line.
        row_index (int): The index of the row (0-indexed).
        row_direction_vector (tuple): The primary direction vector (dx, dy) of the rows.

    Returns:
        tuple: (start_point, end_point, actual_travel_direction_vector)
    """
    coords = np.array(line_string.coords)
    primary_dir = np.array(row_direction_vector)

    if row_index % 2 == 0:  # Even row: travel in primary row_direction_vector
        current_travel_direction = primary_dir
    else:  # Odd row: travel in opposite direction
        current_travel_direction = -primary_dir

    # Project the actual first and last points of the LineString onto the current_travel_direction
    proj_first = np.dot(coords[0], current_travel_direction)
    proj_last = np.dot(coords[-1], current_travel_direction)

    if proj_first <= proj_last:
        # LineString is already ordered in the travel direction
        return coords[0], coords[-1], current_travel_direction
    else:
        # LineString needs to be reversed for travel direction
        return coords[-1], coords[0], current_travel_direction


def genera_traiettorie_svolta(
    trajectories: list[LineString],
    larghezza_mezzo: float,
    lunghezza_mezzo: float,
    raggio_svolta_minimo: float,
    row_direction_vector: tuple
) -> list[LineString]:
    """
    Genera traiettorie di svolta ideali tra file parallele usando curve di Bezier.
    Assume un modello di svolta a "U" (headland turn) dove il veicolo gira alla fine di una fila
    e si allinea per la fila successiva, viaggiando in direzione opposta.

    Args:
        trajectories (list[LineString]): Le traiettorie delle file generate.
        larghezza_mezzo (float): La larghezza di lavoro del mezzo agricolo (usata per spaziatura).
        lunghezza_mezzo (float): La lunghezza del mezzo agricolo (influenza la lunghezza del tratto di svolta).
        raggio_svolta_minimo (float): Il raggio di svolta minimo del mezzo (influenza la curvatura).
        row_direction_vector (tuple): Il vettore di direzione primaria (dx, dy) delle file.

    Returns:
        list[LineString]: Una lista di oggetti LineString che rappresentano le traiettorie di svolta.
    """
    turn_trajectories = []
    if len(trajectories) < 2:
        return []

    for i in range(len(trajectories) - 1):
        current_line = trajectories[i]
        next_line = trajectories[i+1]

        # Get the actual start/end points and travel direction for current and next row
        # P_exit_row_i_start is not used for turn generation, only P_exit_row_i_end
        _, P_exit_row_i_end, dir_i = get_travel_endpoints(current_line, i, row_direction_vector)
        P_entry_row_i_plus_1_start, _, dir_i_plus_1 = get_travel_endpoints(next_line, i + 1, row_direction_vector)

        # P0: Punto di uscita dalla fila corrente
        P0 = P_exit_row_i_end

        # P3: Punto di ingresso nella fila successiva (che verrà percorsa in direzione opposta)
        P3 = P_entry_row_i_plus_1_start

        # Calcola la lunghezza del segmento tangente per le curve di Bezier
        # Questo valore influenza quanto "lontano" si estendono i punti di controllo
        # Una buona euristica è basarla sul raggio di svolta minimo e la larghezza del mezzo
        # per garantire una svolta sufficientemente ampia.
        tangent_length = max(larghezza_mezzo * 1.5, raggio_svolta_minimo * 1.0) # Adjusted heuristic

        # P1: Punto di controllo per l'uscita
        # Si estende lungo la direzione di uscita della fila corrente
        P1 = P0 + tangent_length * dir_i

        # P2: Punto di controllo per l'ingresso
        # Si estende nella direzione opposta a quella di ingresso della fila successiva
        # (perché il veicolo sta girando per entrare in direzione opposta)
        P2 = P3 + tangent_length * (-dir_i_plus_1)

        # Genera la curva di Bezier
        turn_curve = generate_bezier_curve(P0, P1, P2, P3)
        turn_trajectories.append(turn_curve)

    return turn_trajectories


def plot_polygon_and_trajectories(
    original_polygon: Polygon,
    all_path_segments: list[LineString], # Ora accetta una lista combinata di segmenti
    title="Suddivisione con Traiettorie e Svolte"
):
    """
    Visualizza il poligono di input e una lista combinata di traiettorie (file e svolte).

    Args:
        original_polygon (Polygon): Il poligono originale da visualizzare.
        all_path_segments (List[LineString]): Lista ordinata di LineString che rappresentano
                                              l'intero percorso (file e svolte alternate).
        title (str): Titolo del grafico.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Disegna il bordo esterno del poligono originale
    x_poly, y_poly = original_polygon.exterior.xy
    ax.plot(x_poly, y_poly, color='blue', linewidth=2, label='Bordo Poligono')
    ax.fill(x_poly, y_poly, facecolor='lightblue', alpha=0.3)

    # Disegna tutti i segmenti, distinguendo tra file e svolte in base all'indice
    # Si assume che la lista sia interfoliata: fila, svolta, fila, svolta...
    for i, segment in enumerate(all_path_segments):
        x_seg, y_seg = segment.xy
        if i % 2 == 0: # Indice pari: traiettoria di fila
            label = 'Traiettorie File' if i == 0 else "" # Etichetta solo la prima per la legenda
            ax.plot(x_seg, y_seg, color='red', linestyle='--', linewidth=1.5, label=label, alpha=0.7)
        else: # Indice dispari: traiettoria di svolta
            label = 'Traiettorie Svolta' if i == 1 else "" # Etichetta solo la prima per la legenda
            ax.plot(x_seg, y_seg, color='green', linestyle='-', linewidth=2, label=label) # Linea solida per le svolte

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.xlabel("Coordinata X")
    plt.ylabel("Coordinata Y")
    plt.show()


if __name__ == "__main__":
    print("Esecuzione esempio di generazione traiettorie parallele e di svolta...")

    # Parametri del mezzo agricolo
    LARGHEZZA_MEZZO = 0.5 # Corrisponde alla 'larghezza' delle file
    LUNGHEZZA_MEZZO = 3.0 # Lunghezza del veicolo
    RAGGIO_SVOLTA_MINIMO = 2.0 # Raggio di svolta minimo del veicolo

    # Esempio 1: Rettangolo
    rettangolo = Polygon([(0, 0), (6, 0), (6, 2), (0, 2)])
    traiettorie_file1, row_dir1, _ = genera_traiettorie_parallele(rettangolo, larghezza=LARGHEZZA_MEZZO)
    traiettorie_svolta1 = genera_traiettorie_svolta(
        traiettorie_file1, LARGHEZZA_MEZZO, LUNGHEZZA_MEZZO, RAGGIO_SVOLTA_MINIMO, row_dir1
    )
    # Creazione della lista ordinata di segmenti per il percorso completo
    percorso_completo1 = []
    for i in range(len(traiettorie_file1)):
        percorso_completo1.append(traiettorie_file1[i])
        if i < len(traiettorie_svolta1): # Assicura che ci sia una svolta corrispondente
            percorso_completo1.append(traiettorie_svolta1[i])

    plot_polygon_and_trajectories(
        rettangolo, percorso_completo1,
        title="Rettangolo - Percorso Completo"
    )

    # Esempio 2: Trapezio rettangolo
    trapezio = Polygon([(0, 0), (4, 0), (3, 2), (0, 2)])
    traiettorie_file2, row_dir2, _ = genera_traiettorie_parallele(trapezio, larghezza=LARGHEZZA_MEZZO)
    traiettorie_svolta2 = genera_traiettorie_svolta(
        traiettorie_file2, LARGHEZZA_MEZZO, LUNGHEZZA_MEZZO, RAGGIO_SVOLTA_MINIMO, row_dir2
    )
    percorso_completo2 = []
    for i in range(len(traiettorie_file2)):
        percorso_completo2.append(traiettorie_file2[i])
        if i < len(traiettorie_svolta2):
            percorso_completo2.append(traiettorie_svolta2[i])
    plot_polygon_and_trajectories(
        trapezio, percorso_completo2,
        title="Trapezio - Percorso Completo"
    )

    # Esempio 3: Parallelogramma con orientamento manuale
    parallelogramma = Polygon([(0, 0), (4, 0), (5, 2), (1, 2)])
    traiettorie_file3, row_dir3, _ = genera_traiettorie_parallele(parallelogramma, larghezza=LARGHEZZA_MEZZO, orientamento_xy=(2, 1))
    traiettorie_svolta3 = genera_traiettorie_svolta(
        traiettorie_file3, LARGHEZZA_MEZZO, LUNGHEZZA_MEZZO, RAGGIO_SVOLTA_MINIMO, row_dir3
    )
    percorso_completo3 = []
    for i in range(len(traiettorie_file3)):
        percorso_completo3.append(traiettorie_file3[i])
        if i < len(traiettorie_svolta3):
            percorso_completo3.append(traiettorie_svolta3[i])
    plot_polygon_and_trajectories(
        parallelogramma, percorso_completo3,
        title="Parallelogramma - Percorso Completo (Orientamento (2,1))"
    )

    # Esempio 4: Poligono irregolare
    irregolare = Polygon([(0, 0), (5, 1), (4, 4), (1, 3), (0.5, 1.5)])
    traiettorie_file4, row_dir4, _ = genera_traiettorie_parallele(irregolare, larghezza=LARGHEZZA_MEZZO)
    traiettorie_svolta4 = genera_traiettorie_svolta(
        traiettorie_file4, LARGHEZZA_MEZZO, LUNGHEZZA_MEZZO, RAGGIO_SVOLTA_MINIMO, row_dir4
    )
    percorso_completo4 = []
    for i in range(len(traiettorie_file4)):
        percorso_completo4.append(traiettorie_file4[i])
        if i < len(traiettorie_svolta4):
            percorso_completo4.append(traiettorie_svolta4[i])
    plot_polygon_and_trajectories(
        irregolare, percorso_completo4,
        title="Poligono Irregolare - Percorso Completo"
    )

    print("Generazione e visualizzazione delle traiettorie completata.")
