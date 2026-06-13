from shapely.geometry import Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np

class FieldCoveragePlanner:
    """
    Pianificatore per la copertura del campo, genera traiettorie capezzagne
    e visualizza l'area rimanente.
    """
    def __init__(self, area_coordinates, obstacle_coordinates):
        """
        Inizializza il pianificatore con le coordinate dell'area e degli ostacoli.

        Args:
            area_coordinates (list): Lista di tuple (x, y) che definiscono il poligono dell'area.
            obstacle_coordinates (list): Lista di liste di tuple (x, y) per ogni ostacolo.
        """
        self.area = Polygon(area_coordinates)
        self.obstacles = [Polygon(coords) for coords in obstacle_coordinates]
        self.capezzagne = []
        self.traiettorie_lavoro = []
        self.campo_rimanente = None

    def genera_capezzagne_e_traiettorie(self, ampiezza_lavoro, numero_passate):
        """
        Genera le traiettorie capezzagne concentriche e le traiettorie di lavoro interne,
        evitando gli ostacoli.

        Args:
            ampiezza_lavoro (float): L'ampiezza di lavoro del mezzo (distanza tra le traiettorie).
            numero_passate (int): Il numero di passate di capezzagna da generare.
        """
        self.capezzagne = []
        self.traiettorie_lavoro = []
        self.campo_rimanente = self.area

        # Inizialmente, rimuovi gli ostacoli dall'area totale per definire il campo lavorabile
        for obstacle in self.obstacles:
            self.campo_rimanente = self.campo_rimanente.difference(obstacle)

        # Assicurati che il campo rimanente sia un poligono valido dopo la sottrazione
        if self.campo_rimanente.is_empty or not self.campo_rimanente.is_valid:
            print("L'area è completamente occupata dagli ostacoli o non è valida.")
            return

        current_working_area = self.campo_rimanente

        for i in range(numero_passate):
            # Calcola il poligono interno per la prossima capezzagna
            # Usiamo un buffer negativo per restringere l'area
            buffered_area = current_working_area.buffer(-ampiezza_lavoro, join_style=2, mitre_limit=5.0) # join_style=2 per angoli smussati

            # Gestisci il caso in cui il buffer risulti in una geometria vuota o non valida
            if buffered_area.is_empty or not buffered_area.is_valid:
                print(f"Buffer {i+1} ha prodotto un'area vuota o non valida. Fine generazione capezzagne.")
                break

            # La capezzagna è l'area tra il bordo corrente e il bordo bufferizzato
            capezzagna_segment = current_working_area.difference(buffered_area)

            # Aggiungi le parti valide della capezzagna
            if capezzagna_segment.geom_type == 'Polygon' and not capezzagna_segment.is_empty:
                self.capezzagne.append(capezzagna_segment)
            elif capezzagna_segment.geom_type == 'MultiPolygon':
                for poly in capezzagna_segment.geoms:
                    if poly.geom_type == 'Polygon' and not poly.is_empty:
                        self.capezzagne.append(poly)
            elif capezzagna_segment.geom_type == 'GeometryCollection':
                for geom in capezzagna_segment.geoms:
                    if geom.geom_type == 'Polygon' and not geom.is_empty:
                        self.capezzagne.append(geom)

            # Aggiorna l'area di lavoro corrente per la prossima iterazione
            current_working_area = buffered_area

        # Il campo rimanente è l'ultima area di lavoro dopo aver generato tutte le capezzagne
        self.campo_rimanente = current_working_area

        # Genera le traiettorie di lavoro all'interno del campo rimanente (per semplicità, linee verticali)
        # Questo è un esempio; in un'applicazione reale, la logica sarebbe più complessa
        if self.campo_rimanente and not self.campo_rimanente.is_empty:
            minx, miny, maxx, maxy = self.campo_rimanente.bounds
            num_lines = int((maxx - minx) / ampiezza_lavoro) + 1 # Aggiungi 1 per coprire l'intera larghezza

            for k in range(num_lines):
                x_coord = minx + k * ampiezza_lavoro
                line = LineString([(x_coord, miny), (x_coord, maxy)])
                
                # Interseca la linea con il campo rimanente per ottenere i segmenti validi
                intersection = self.campo_rimanente.intersection(line)
                
                if intersection.geom_type == 'LineString' and not intersection.is_empty:
                    self.traiettorie_lavoro.append(intersection)
                elif intersection.geom_type == 'MultiLineString':
                    for segment in intersection.geoms:
                        if segment.geom_type == 'LineString' and not segment.is_empty:
                            self.traiettorie_lavoro.append(segment)

    def get_campo_rimanente_coords(self):
        """
        Restituisce le coordinate del poligono del campo rimanente.
        Utile per l'input al prossimo algoritmo.

        Returns:
            list: Una lista di coordinate (x, y) per il campo rimanente.
                  Se il campo rimanente è un MultiPolygon, restituisce le coordinate
                  di tutti i poligoni.
        """
        if self.campo_rimanente.geom_type == 'Polygon':
            return list(self.campo_rimanente.exterior.coords)
        elif self.campo_rimanente.geom_type == 'MultiPolygon':
            all_coords = []
            for poly in self.campo_rimanente.geoms:
                if poly.geom_type == 'Polygon':
                    all_coords.extend(list(poly.exterior.coords))
            return all_coords
        return []

# --- Funzioni di Visualizzazione ---
def plot_polygon(ax, polygon, facecolor='blue', edgecolor='black', alpha=0.5, linestyle='-', linewidth=1, label=None):
    """Disegna un poligono o MultiPolygon sul grafico."""
    if polygon.geom_type == 'Polygon':
        x, y = polygon.exterior.xy
        ax.fill(x, y, facecolor=facecolor, alpha=alpha)
        ax.plot(x, y, color=edgecolor, linestyle=linestyle, linewidth=linewidth, label=label)
        for interior in polygon.interiors:  # Gestisce i buchi
            x_i, y_i = interior.xy
            ax.fill(x_i, y_i, facecolor='white', alpha=1)  # Riempie i buchi di bianco
            ax.plot(x_i, y_i, color=edgecolor, linestyle=linestyle, linewidth=linewidth)  # Bordo dei buchi
    elif polygon.geom_type == 'MultiPolygon':
        # Applica l'etichetta solo al primo poligono per evitare duplicati nella legenda
        first_poly_plotted = False
        for poly in polygon.geoms:
            if poly.geom_type == 'Polygon':
                if not first_poly_plotted and label:
                    plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label=label)
                    first_poly_plotted = True
                else:
                    plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
    elif polygon.geom_type == 'GeometryCollection':
        first_poly_plotted = False
        for geom in polygon.geoms:
            if geom.geom_type == 'Polygon':
                if not first_poly_plotted and label:
                    plot_polygon(ax, geom, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle, linewidth=linewidth, label=label)
                    first_poly_plotted = True
                else:
                    plot_polygon(ax, geom, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle, linewidth=linewidth)


def plot_lines(ax, lines, color='black', linestyle='-', linewidth=1, label=None):
    """Disegna una lista di LineString sul grafico."""
    first_line_plotted = False
    for line in lines:
        if line.geom_type == 'LineString':
            x, y = line.xy
            if not first_line_plotted and label:
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
                first_line_plotted = True
            else:
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
        elif line.geom_type == 'MultiLineString':
            for seg in line.geoms:
                if seg.geom_type == 'LineString':
                    x, y = seg.xy
                    if not first_line_plotted and label:
                        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
                        first_line_plotted = True
                    else:
                        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)


# --- Esempi di Test ---
if __name__ == '__main__':
    # --- Esempio 1: Campo rettangolare con ostacolo quadrato ---
    print("--- Esempio 1: Campo rettangolare con ostacolo quadrato ---")
    area_coords_1 = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    obstacle_coords_1 = [[(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]]
    ampiezza_lavoro_1 = 1.0
    numero_passate_1 = 2

    planner1 = FieldCoveragePlanner(area_coords_1, obstacle_coords_1)
    planner1.genera_capezzagne_e_traiettorie(ampiezza_lavoro_1, numero_passate_1)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_title(f"Campo Rettangolare - {numero_passate_1} Capezzagne (Ampiezza: {ampiezza_lavoro_1})")
    plot_polygon(ax1, planner1.area, facecolor='lightgreen', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in planner1.obstacles:
        plot_polygon(ax1, obstacle, facecolor='red', edgecolor='black', alpha=1.0, label='Ostacoli')

    # Disegna le capezzagne
    cmap_capezzagne = plt.colormaps.get_cmap('Oranges')
    for i, capezzagna in enumerate(planner1.capezzagne):
        plot_polygon(ax1, capezzagna, facecolor=cmap_capezzagne(i / numero_passate_1), edgecolor='darkorange', alpha=0.6, linewidth=1.5, label=f'Capezzagna {i+1}' if i == 0 else None) # Label solo per la prima capezzagna

    # Disegna il campo rimanente (nuovo bordo del campo)
    if planner1.campo_rimanente and not planner1.campo_rimanente.is_empty:
        plot_polygon(ax1, planner1.campo_rimanente, facecolor='skyblue', edgecolor='blue', alpha=0.5, linewidth=2, linestyle='-', label='Campo Rimanente')

    # Disegna le traiettorie di lavoro tratteggiate
    plot_lines(ax1, planner1.traiettorie_lavoro, color='blue', linestyle='--', linewidth=0.8, label='Traiettorie di Lavoro')

    ax1.set_aspect('equal', adjustable='box')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True)
    plt.show()

    print("Coordinate del campo rimanente (Esempio 1):", planner1.get_campo_rimanente_coords())
    print("-" * 50)

    # --- Esempio 2: Campo a forma di L con ostacoli multipli ---
    print("--- Esempio 2: Campo a forma di L con ostacoli multipli ---")
    area_coords_2 = [
        (0,0), (10,0), (10,5), (5,5), (5,10), (0,10), (0,0)
    ]
    obstacle_coords_2 = [
        [(2,2), (3,2), (3,3), (2,3), (2,2)],
        [(7,1), (8,1), (8,2), (7,2), (7,1)],
        [(6,7), (7,7), (7,8), (6,8), (6,7)]
    ]
    ampiezza_lavoro_2 = 1.5
    numero_passate_2 = 1

    planner2 = FieldCoveragePlanner(area_coords_2, obstacle_coords_2)
    planner2.genera_capezzagne_e_traiettorie(ampiezza_lavoro_2, numero_passate_2)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.set_title(f"Campo a L - {numero_passate_2} Capezzagna (Ampiezza: {ampiezza_lavoro_2})")
    plot_polygon(ax2, planner2.area, facecolor='lightgreen', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in planner2.obstacles:
        plot_polygon(ax2, obstacle, facecolor='red', edgecolor='black', alpha=1.0, label='Ostacoli')

    cmap_capezzagne = plt.colormaps.get_cmap('Oranges')
    for i, capezzagna in enumerate(planner2.capezzagne):
        plot_polygon(ax2, capezzagna, facecolor=cmap_capezzagne(i / numero_passate_2), edgecolor='darkorange', alpha=0.6, linewidth=1.5, label=f'Capezzagna {i+1}' if i == 0 else None)

    if planner2.campo_rimanente and not planner2.campo_rimanente.is_empty:
        plot_polygon(ax2, planner2.campo_rimanente, facecolor='skyblue', edgecolor='blue', alpha=0.5, linewidth=2, linestyle='-', label='Campo Rimanente')

    plot_lines(ax2, planner2.traiettorie_lavoro, color='blue', linestyle='--', linewidth=0.8, label='Traiettorie di Lavoro')

    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True)
    plt.show()

    print("Coordinate del campo rimanente (Esempio 2):", planner2.get_campo_rimanente_coords())
    print("-" * 50)

    # --- Esempio 3: Campo irregolare complesso con ostacoli di forma diversa ---
    print("--- Esempio 3: Campo irregolare complesso con ostacoli di forma diversa ---")
    area_coords_3 = [
        (0,0), (15,0), (15,12), (10,12), (10,4), (5,4), (5,12), (0,12), (0,0)
    ]
    obstacle_coords_3 = [
        # Ostacolo a L
        [(3,2), (6,2), (6,3), (4,3), (4,5), (3,5), (3,2)],
        # Ostacolo a forma di diamante
        [(8,7), (9,8), (8,9), (7,8), (8,7)],
        # Ostacolo irregolare
        [(12,1), (13,2), (12.5,4), (11.5,3), (12,1)]
    ]
    ampiezza_lavoro_3 = 1.2
    numero_passate_3 = 3

    planner3 = FieldCoveragePlanner(area_coords_3, obstacle_coords_3)
    planner3.genera_capezzagne_e_traiettorie(ampiezza_lavoro_3, numero_passate_3)

    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.set_title(f"Campo Irregolare - {numero_passate_3} Capezzagne (Ampiezza: {ampiezza_lavoro_3})")
    plot_polygon(ax3, planner3.area, facecolor='lightgreen', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in planner3.obstacles:
        plot_polygon(ax3, obstacle, facecolor='red', edgecolor='black', alpha=1.0, label='Ostacoli')

    cmap_capezzagne = plt.colormaps.get_cmap('Oranges')
    for i, capezzagna in enumerate(planner3.capezzagne):
        plot_polygon(ax3, capezzagna, facecolor=cmap_capezzagne(i / numero_passate_3), edgecolor='darkorange', alpha=0.6, linewidth=1.5, label=f'Capezzagna {i+1}' if i == 0 else None)

    if planner3.campo_rimanente and not planner3.campo_rimanente.is_empty:
        plot_polygon(ax3, planner3.campo_rimanente, facecolor='skyblue', edgecolor='blue', alpha=0.5, linewidth=2, linestyle='-', label='Campo Rimanente')

    plot_lines(ax3, planner3.traiettorie_lavoro, color='blue', linestyle='--', linewidth=0.8, label='Traiettorie di Lavoro')

    ax3.set_aspect('equal', adjustable='box')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax3.grid(True)
    plt.show()

    print("Coordinate del campo rimanente (Esempio 3):", planner3.get_campo_rimanente_coords())
    print("-" * 50)
