from shapely.geometry import Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt

class HeadlandCalculator:
    def __init__(self, area_coordinates, obstacle_coordinates):
        """
        Inizializza il calcolatore delle capezzagne.

        Args:
            area_coordinates (list): Lista di tuple (x, y) che definiscono il poligono dell'area.
            obstacle_coordinates (list): Lista di liste di tuple (x, y) per ogni ostacolo.
        """
        self.area = Polygon(area_coordinates)
        self.obstacles = [Polygon(coords) for coords in obstacle_coordinates]
        self.effective_area = self.area # Area iniziale, da cui verranno sottratti gli ostacoli
        self.obstacles_to_avoid = [] # Ostacoli che intersecano il bordo dell'area

        # Pre-processa gli ostacoli per definire l'area effettiva e gli ostacoli da aggirare
        for obstacle in self.obstacles:
            if self.area.intersects(obstacle):
                self.effective_area = self.effective_area.difference(obstacle)
                # Se l'ostacolo è vicino al bordo (o lo interseca), lo consideriamo per l'aggiramento
                # Usiamo un piccolo buffer per catturare ostacoli "vicini" al bordo
                if self.area.exterior.buffer(0.01).intersects(obstacle):
                    self.obstacles_to_avoid.append(obstacle)

    def calculate_headlands(self, work_width, num_headlands=None):
        """
        Calcola le capezzagne di bordo campo e aggiorna l'area interna.

        Args:
            work_width (float): Ampiezza di lavoro dell'attrezzo.
            num_headlands (int, optional): Numero di capezzagne da generare.
                                           Se None o non specificato, userà un valore minimo predefinito (es. 2).

        Returns:
            tuple: (headland_trajectories, inner_area_coords)
                    - headland_trajectories: Lista di LineString che rappresentano le traiettorie delle capezzagne.
                    - inner_area_coords: Lista di tuple (x, y) che definiscono il nuovo poligono interno.
        """
        # INIZIO LOGICA DI CONTROLLO E ASSEGNAZIONE DI num_headlands_effective
        if num_headlands is None:
            num_headlands_effective = 1 # Valore predefinito se non specificato
            print(f"Attenzione: num_headlands non specificato o None. Usando il valore predefinito: {num_headlands_effective}")
        elif not isinstance(num_headlands, int) or num_headlands < 0:
            # Opzionale: gestire casi di input non validi (es. float, negativo)
            num_headlands_effective = 1
            print(f"Attenzione: num_headlands '{num_headlands}' non è un intero valido. Usando il valore predefinito: {num_headlands_effective}")
        else:
            num_headlands_effective = num_headlands
        # FINE LOGICA DI CONTROLLO

        headland_trajectories = []
        
        # Calcola l'area interna finale dopo aver rimosso tutte le capezzagne
        # Questo serve per definire il nuovo "bordo campo" interno
        total_offset_for_inner_area = -num_headlands_effective * work_width # CORRETTO: USA num_headlands_effective
        final_inner_area = self.effective_area.buffer(total_offset_for_inner_area, join_style=3, mitre_limit=5.0)

        # Assicurati che l'area interna finale sia un poligono valido, altrimenti è vuota
        if final_inner_area.is_empty or not final_inner_area.is_valid:
            final_inner_area = Polygon() # Rappresenta un'area vuota se si è collassata

        # Genera le traiettorie delle capezzagne
        for i in range(num_headlands_effective): # CORRETTO: USA num_headlands_effective
            # La traiettoria per la capezzagna 'i' è centrata a (i + 0.5) * work_width
            # dal bordo esterno originale dell'area effettiva.
            offset_for_trajectory = -(i + 0.5) * work_width
            
            # Crea un poligono temporaneo facendo il buffer dell'area effettiva originale.
            # Il bordo di questo poligono temporaneo sarà la traiettoria desiderata.
            temp_trajectory_poly = self.effective_area.buffer(offset_for_trajectory, join_style=3, mitre_limit=5.0)

            if temp_trajectory_poly.is_empty or not temp_trajectory_poly.is_valid:
                # Se il buffer per questa traiettoria collassa, interrompi la generazione delle capezzagne successive
                break

            # Il contorno (exterior) di questo poligono temporaneo è la traiettoria desiderata
            if temp_trajectory_poly.geom_type == 'Polygon':
                headland_trajectories.append(LineString(temp_trajectory_poly.exterior.coords))
            elif temp_trajectory_poly.geom_type == 'MultiPolygon':
                # Se l'offset crea più parti, aggiungi il contorno di ogni parte
                for poly_part in temp_trajectory_poly.geoms:
                    if poly_part.geom_type == 'Polygon':
                        headland_trajectories.append(LineString(poly_part.exterior.coords))
            # GeometryCollection non dovrebbe essere il risultato di un buffer, quindi non è gestito esplicitamente qui.

        # Determina i nuovi vertici interni basati sull'area interna finale calcolata
        inner_area_coords = []
        if final_inner_area.geom_type == 'Polygon':
            inner_area_coords = list(final_inner_area.exterior.coords)
        elif final_inner_area.geom_type == 'MultiPolygon':
            # Se l'area interna è un MultiPolygon, prendi il poligono più grande
            # (assumendo che sia la parte principale dell'area di lavoro)
            max_area = 0
            for poly in final_inner_area.geoms:
                if poly.geom_type == 'Polygon' and poly.area > max_area:
                    max_area = poly.area
                    inner_area_coords = list(poly.exterior.coords)
        elif final_inner_area.geom_type == 'GeometryCollection':
            # Cerca il poligono più grande all'interno della GeometryCollection
            max_area = 0
            for geom in final_inner_area.geoms:
                if geom.geom_type == 'Polygon' and geom.area > max_area:
                    max_area = geom.area
                    inner_area_coords = list(geom.exterior.coords)

        return headland_trajectories, inner_area_coords

# --- Funzioni di Visualizzazione (rimangono invariate) ---
def plot_polygon(ax, polygon, facecolor='blue', edgecolor='black', alpha=0.5, label=None):
    """Disegna un poligono sul grafico con un colore di riempimento e un bordo."""
    if polygon.geom_type == 'Polygon':
        x, y = polygon.exterior.xy
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, label=label)
        ax.plot(x, y, color=edgecolor, linewidth=1) # Disegna il contorno
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.fill(x, y, facecolor='white', alpha=1) # I buchi sono bianchi
            ax.plot(x, y, color=edgecolor, linewidth=1) # Disegna il contorno del buco
    elif polygon.geom_type == 'MultiPolygon':
        for i, poly in enumerate(polygon.geoms):
            # Passa il label solo al primo poligono se è un MultiPolygon per evitare duplicati nella legenda
            plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, label=label if i == 0 else None)

def plot_segments(ax, segments, color='black', linewidth=2, linestyle='-', label=None):
    """Disegna una lista di segmenti di linea sul grafico."""
    for segment in segments:
        x, y = segment.xy
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, label=label)

def plot_points(ax, points, color='red', marker='o', markersize=None, label=None, linestyle='None'):
    """Disegna una lista di punti sul grafico."""
    if not points: # Gestisce il caso di lista vuota
        return
    x, y = zip(*points)
    if markersize is not None:
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, markersize=markersize, label=label)
    else:
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, label=label)

if __name__ == '__main__':
    # Esempio con num_headlands non specificato (sarà 2) - aggiunto per mostrare il default
    print("\n--- Esempio con num_headlands non specificato (default 2) ---")
    area_coords_default = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    obstacle_coords_default = []
    work_width_default = 0.8

    calculator_default = HeadlandCalculator(area_coords_default, obstacle_coords_default)
    headland_trajectories_default, inner_area_coords_default = calculator_default.calculate_headlands(work_width_default)

    fig_default, ax_default = plt.subplots(figsize=(8, 8))
    plot_polygon(ax_default, Polygon(area_coords_default), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for i, trajectory in enumerate(headland_trajectories_default):
        plot_segments(ax_default, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')
    if inner_area_coords_default:
        plot_polygon(ax_default, Polygon(inner_area_coords_default), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')
    ax_default.set_aspect('equal', adjustable='box')
    ax_default.set_title(f"Calcolo Capezzagne (Default) - {len(headland_trajectories_default)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate (default): {len(headland_trajectories_default)}")
    print(f"Vertici area interna finale (default): {inner_area_coords_default[:5]}...")

    # Esempio con num_headlands esplicitamente a 0 - aggiunto per mostrare la gestione di 0
    print("\n--- Esempio con num_headlands esplicitamente a 0 ---")
    area_coords_zero = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    obstacle_coords_zero = []
    work_width_zero = 0.8
    num_headlands_zero = 0

    calculator_zero = HeadlandCalculator(area_coords_zero, obstacle_coords_zero)
    headland_trajectories_zero, inner_area_coords_zero = calculator_zero.calculate_headlands(work_width_zero, num_headlands_zero)

    fig_zero, ax_zero = plt.subplots(figsize=(8, 8))
    plot_polygon(ax_zero, Polygon(area_coords_zero), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for i, trajectory in enumerate(headland_trajectories_zero):
        plot_segments(ax_zero, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')
    if inner_area_coords_zero:
        plot_polygon(ax_zero, Polygon(inner_area_coords_zero), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')
    ax_zero.set_aspect('equal', adjustable='box')
    ax_zero.set_title(f"Calcolo Capezzagne (Zero) - {len(headland_trajectories_zero)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate (zero): {len(headland_trajectories_zero)}")
    print(f"Vertici area interna finale (zero): {inner_area_coords_zero[:5]}...")

    # --- Esempio 1: Poligono semplice con ostacolo interno e sul bordo --- (Primo blocco di esempi)
    print("--- Esempio 1: Poligono semplice con ostacolo interno e sul bordo ---")
    area_coords_1 = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    obstacle_coords_1 = [
        [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)],   # Ostacolo interno
        [(9, 0), (10, 0), (10, 1), (9, 1), (9, 0)]   # Ostacolo sul bordo (angolo)
    ]
    work_width_1 = 0.8
    num_headlands_1 = 2

    calculator1 = HeadlandCalculator(area_coords_1, obstacle_coords_1)
    headland_trajectories_1, inner_area_coords_1 = calculator1.calculate_headlands(work_width_1, num_headlands_1)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax1, Polygon(area_coords_1), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_1:
        plot_polygon(ax1, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_1):
        plot_segments(ax1, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_1:
        plot_polygon(ax1, Polygon(inner_area_coords_1), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(f"Calcolo Capezzagne (Esempio 1) - {len(headland_trajectories_1)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_1)}")
    print(f"Vertici area interna finale: {inner_area_coords_1[:5]}...") # Mostra solo i primi 5 per brevità

    # --- Esempio 2: Area a forma di L con ostacolo interno ---
    print("\n--- Esempio 2: Area a forma di L con ostacolo interno ---")
    area_coords_2 = [
        (0,0), (10,0), (10,5), (5,5), (5,10), (0,10), (0,0)
    ]
    obstacle_coords_2 = [
        [(2,2), (3,2), (3,3), (2,3), (2,2)] # Piccolo ostacolo interno
    ]
    work_width_2 = 0.7
    num_headlands_2 = 3

    calculator2 = HeadlandCalculator(area_coords_2, obstacle_coords_2)
    headland_trajectories_2, inner_area_coords_2 = calculator2.calculate_headlands(work_width_2, num_headlands_2)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax2, Polygon(area_coords_2), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_2:
        plot_polygon(ax2, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_2):
        plot_segments(ax2, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_2:
        plot_polygon(ax2, Polygon(inner_area_coords_2), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(f"Calcolo Capezzagne (Esempio 2) - {len(headland_trajectories_2)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_2)}")
    print(f"Vertici area interna finale: {inner_area_coords_2[:5]}...")

    # --- Esempio 3: Area a forma di C con ostacolo sul "braccio" ---
    print("\n--- Esempio 3: Area a forma di C con ostacolo sul 'braccio' ---")
    area_coords_3 = [
        (0,0), (12,0), (12,10), (10,10), (10,2), (2,2), (2,10), (0,10), (0,0)
    ]
    obstacle_coords_3 = [
        [(3,4), (4,4), (4,6), (3,6), (3,4)] # Ostacolo sul braccio sinistro della C
    ]
    work_width_3 = 0.9
    num_headlands_3 = 2

    calculator3 = HeadlandCalculator(area_coords_3, obstacle_coords_3)
    headland_trajectories_3, inner_area_coords_3 = calculator3.calculate_headlands(work_width_3, num_headlands_3)

    fig3, ax3 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax3, Polygon(area_coords_3), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_3:
        plot_polygon(ax3, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_3):
        plot_segments(ax3, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_3:
        plot_polygon(ax3, Polygon(inner_area_coords_3), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title(f"Calcolo Capezzagne (Esempio 3) - {len(headland_trajectories_3)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_3)}")
    print(f"Vertici area interna finale: {inner_area_coords_3[:5]}...")

    # --- Esempio 4: Area irregolare con ostacoli multipli e sul bordo ---
    print("\n--- Esempio 4: Area irregolare con ostacoli multipli e sul bordo ---")
    area_coords_4 = [
        (0,0), (15,0), (15,12), (10,12), (10,4), (5,4), (5,12), (0,12), (0,0)
    ]
    obstacle_coords_4 = [
        [(3,2), (6,2), (6,3), (4,3), (4,5), (3,5), (3,2)], # Ostacolo a L interno
        [(8,7), (9,8), (8,9), (7,8), (8,7)], # Ostacolo a forma di diamante interno
        [(14,1), (15,1), (15,3), (14,3), (14,1)] # Ostacolo sul bordo destro
    ]
    work_width_4 = 1.0
    num_headlands_4 = 2

    calculator4 = HeadlandCalculator(area_coords_4, obstacle_coords_4)
    headland_trajectories_4, inner_area_coords_4 = calculator4.calculate_headlands(work_width_4, num_headlands_4)

    fig4, ax4 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax4, Polygon(area_coords_4), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_4:
        plot_polygon(ax4, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_4):
        plot_segments(ax4, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_4:
        plot_polygon(ax4, Polygon(inner_area_coords_4), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title(f"Calcolo Capezzagne (Esempio 4) - {len(headland_trajectories_4)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_4)}")
    print(f"Vertici area interna finale: {inner_area_coords_4[:5]}...")

    # --- Esempio 5: Area a forma di U con ostacolo che chiude parzialmente il "buco" ---
    print("\n--- Esempio 5: Area a forma di U con ostacolo che chiude parzialmente il 'buco' ---")
    area_coords_5 = [
        (0,0), (10,0), (10,10), (8,10), (8,2), (2,2), (2,10), (0,10), (0,0)
    ]
    obstacle_coords_5 = [
        [(4,0), (6,0), (6,2), (4,2), (4,0)] # Ostacolo che chiude parzialmente la base della U
    ]
    work_width_5 = 0.5
    num_headlands_5 = 3

    calculator5 = HeadlandCalculator(area_coords_5, obstacle_coords_5)
    headland_trajectories_5, inner_area_coords_5 = calculator5.calculate_headlands(work_width_5, num_headlands_5)

    fig5, ax5 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax5, Polygon(area_coords_5), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_5:
        plot_polygon(ax5, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_5):
        plot_segments(ax5, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_5:
        plot_polygon(ax5, Polygon(inner_area_coords_5), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax5.set_aspect('equal', adjustable='box')
    ax5.set_title(f"Calcolo Capezzagne (Esempio 5) - {len(headland_trajectories_5)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_5)}")
    print(f"Vertici area interna finale: {inner_area_coords_5[:5]}...")

    # --- NUOVO Esempio 6: Campo grande e irregolare con ostacoli multipli ---
    print("\n--- Esempio 6: Campo grande e irregolare con ostacoli multipli ---")
    area_coords_6 = [
        (0,0), (20,0), (22,5), (20,15), (15,18), (10,17), (5,16), (2,10), (0,5), (0,0)
    ]
    obstacle_coords_6 = [
        [(3,2), (5,2), (5,4), (3,4), (3,2)], # Ostacolo interno 1
        [(12,3), (14,3), (14,5), (12,5), (12,3)], # Ostacolo interno 2
        [(18,10), (19,10), (19,12), (18,12), (18,10)], # Ostacolo interno 3
        [(0,7), (1,7), (1,8), (0,8), (0,7)], # Ostacolo sul bordo sinistro
        [(19,0), (20,0), (20,1), (19,1), (19,0)] # Ostacolo sul bordo in basso a destra
    ]
    work_width_6 = 1.2
    num_headlands_6 = 3

    calculator6 = HeadlandCalculator(area_coords_6, obstacle_coords_6)
    headland_trajectories_6, inner_area_coords_6 = calculator6.calculate_headlands(work_width_6, num_headlands_6)

    fig6, ax6 = plt.subplots(figsize=(10, 10))
    plot_polygon(ax6, Polygon(area_coords_6), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_6:
        plot_polygon(ax6, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_6):
        plot_segments(ax6, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_6:
        plot_polygon(ax6, Polygon(inner_area_coords_6), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax6.set_aspect('equal', adjustable='box')
    ax6.set_title(f"Calcolo Capezzagne (Esempio 6) - {len(headland_trajectories_6)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_6)}")
    print(f"Vertici area interna finale: {inner_area_coords_6[:5]}...")

    # --- NUOVO Esempio 7: Campo con "buco a ciambella" e ostacoli sul bordo ---
    print("\n--- Esempio 7: Campo con 'buco a ciambella' e ostacoli sul bordo ---")
    outer_boundary = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
    inner_hole = [(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)]

    # CORREZIONE QUI: Passa solo il bordo esterno come area e l'interno come ostacolo
    area_coords_7_corrected = outer_boundary
    obstacle_coords_7_corrected = [
        inner_hole, # Il buco interno diventa un ostacolo
        [(1,18), (3,18), (3,19), (1,19), (1,18)], # Ostacolo sul bordo in alto a sinistra
        [(17,1), (19,1), (19,3), (17,3), (17,1)]   # Ostacolo sul bordo in basso a destra
    ]
    work_width_7 = 1.0
    num_headlands_7 = 4

    # Usa le coordinate corrette per l'inizializzazione
    calculator7 = HeadlandCalculator(area_coords_7_corrected, obstacle_coords_7_corrected)
    headland_trajectories_7, inner_area_coords_7 = calculator7.calculate_headlands(work_width_7, num_headlands_7)

    fig7, ax7 = plt.subplots(figsize=(10, 10))
    # Quando tracci l'area originale, usa il Polygon che ha anche il buco per la visualizzazione
    plot_polygon(ax7, Polygon(outer_boundary, [inner_hole]), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_7_corrected: # Traccia tutti gli ostacoli, incluso il buco
        plot_polygon(ax7, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_7):
        plot_segments(ax7, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_7:
        plot_polygon(ax7, Polygon(inner_area_coords_7), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax7.set_aspect('equal', adjustable='box')
    ax7.set_title(f"Calcolo Capezzagne (Esempio 7) - {len(headland_trajectories_7)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_7)}")
    print(f"Vertici area interna finale: {inner_area_coords_7[:5]}...")

    # --- NUOVO Esempio 8: Campo multi-lobato con molti ostacoli ---
    print("\n--- Esempio 8: Campo multi-lobato con molti ostacoli ---")
    area_coords_8 = [
        (0,0), (25,0), (25,5), (20,5), (20,10), (25,10), (25,15), (15,15),
        (15,10), (10,10), (10,15), (0,15), (0,0)
    ]
    obstacle_coords_8 = [
        [(2,2), (4,2), (4,4), (2,4), (2,2)], # Ostacolo 1
        [(7,1), (9,1), (9,3), (7,3), (7,1)], # Ostacolo 2
        [(12,2), (14,2), (14,4), (12,4), (12,2)], # Ostacolo 3
        [(22,2), (24,2), (24,4), (22,4), (22,2)], # Ostacolo 4 (sul bordo)
        [(1,12), (3,12), (3,14), (1,14), (1,12)], # Ostacolo 5 (sul bordo)
        [(17,12), (19,12), (19,14), (17,14), (17,12)], # Ostacolo 6 (sul bordo)
        [(12,7), (13,7), (13,8), (12,8), (12,7)] # Ostacolo centrale
    ]
    work_width_8 = 0.7
    num_headlands_8 = 5

    calculator8 = HeadlandCalculator(area_coords_8, obstacle_coords_8)
    headland_trajectories_8, inner_area_coords_8 = calculator8.calculate_headlands(work_width_8, num_headlands_8)

    fig8, ax8 = plt.subplots(figsize=(12, 12))
    plot_polygon(ax8, Polygon(area_coords_8), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_8:
        plot_polygon(ax8, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    
    for i, trajectory in enumerate(headland_trajectories_8):
        plot_segments(ax8, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')

    if inner_area_coords_8:
        plot_polygon(ax8, Polygon(inner_area_coords_8), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')

    ax8.set_aspect('equal', adjustable='box')
    ax8.set_title(f"Calcolo Capezzagne (Esempio 8) - {len(headland_trajectories_8)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Numero di capezzagne generate: {len(headland_trajectories_8)}")
    print(f"Vertici area interna finale: {inner_area_coords_8[:5]}...")

    # --- Esempio 1: Campo rettangolare con ostacoli irregolari --- (Secondo blocco di esempi)
    print("\n--- Esempio 1: Campo rettangolare con ostacoli irregolari ---")
    area_coords_1_b2 = [(0, 0), (50, 0), (50, 30), (0, 30), (0, 0)]
    obstacle_coords_1_b2 = [
        [(10, 5), (15, 7), (12, 12), (8, 10), (10, 5)],    # Ostacolo irregolare 1
        [(30, 18), (35, 20), (32, 25), (28, 23), (30, 18)],    # Ostacolo irregolare 2
        [(48, 28), (49, 27), (49, 29), (48, 29), (48, 28)], # Ostacolo sul bordo
        [(1,1), (2,1), (2,2), (1,2), (1,1)]
    ]
    work_width_1_b2 = 2.0
    num_headlands_1_b2 = -4 # Questo innesca il messaggio di avviso e il default a 2

    calculator1_b2 = HeadlandCalculator(area_coords_1_b2, obstacle_coords_1_b2)
    headland_trajectories_1_b2, inner_area_coords_1_b2 = calculator1_b2.calculate_headlands(work_width_1_b2, num_headlands_1_b2)

    fig1_b2, ax1_b2 = plt.subplots(figsize=(10, 6))
    plot_polygon(ax1_b2, Polygon(area_coords_1_b2), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_1_b2:
        plot_polygon(ax1_b2, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    for i, trajectory in enumerate(headland_trajectories_1_b2):
        plot_segments(ax1_b2, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')
    if inner_area_coords_1_b2:
        plot_polygon(ax1_b2, Polygon(inner_area_coords_1_b2), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')
    ax1_b2.set_aspect('equal', adjustable='box')
    ax1_b2.set_title(f"Campo Rettangolare con Ostacoli Irregolari - {len(headland_trajectories_1_b2)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Esempio 2: Campo a forma di L con isola centrale --- (Secondo blocco di esempi)
    print("\n--- Esempio 2: Campo a forma di L con isola centrale ---")
    area_coords_2_b2 = [(0, 0), (40, 0), (40, 20), (20, 20), (20, 40), (0, 40), (0, 0)]
    obstacle_coords_2_b2 = [
        [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)],    # Isola centrale
        [(38, 1), (39, 1), (39, 2), (38, 2), (38, 1)], # Ostacolo sul bordo
        [(1,38), (2,38), (2,39), (1,39), (1,38)]
    ]
    work_width_2_b2 = 3.0
    num_headlands_2_b2 = 2

    calculator2_b2 = HeadlandCalculator(area_coords_2_b2, obstacle_coords_2_b2)
    headland_trajectories_2_b2, inner_area_coords_2_b2 = calculator2_b2.calculate_headlands(work_width_2_b2, num_headlands_2_b2)

    fig2_b2, ax2_b2 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax2_b2, Polygon(area_coords_2_b2), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_2_b2:
        plot_polygon(ax2_b2, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    for i, trajectory in enumerate(headland_trajectories_2_b2):
        plot_segments(ax2_b2, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')
    if inner_area_coords_2_b2:
        plot_polygon(ax2_b2, Polygon(inner_area_coords_2_b2), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')
    ax2_b2.set_aspect('equal', adjustable='box')
    ax2_b2.set_title(f"Campo a Forma di L con Isola - {len(headland_trajectories_2_b2)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Esempio 3: Campo con forma a 'ciambella' e ostacoli sul bordo --- (Secondo blocco di esempi)
    print("\n--- Esempio 3: Campo con forma a 'ciambella' e ostacoli sul bordo ---")
    outer_boundary_b2 = [(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]
    inner_hole_b2 = [(10, 10), (40, 10), (40, 40), (10, 40), (10, 10)]
    
    # CORREZIONE QUI: Passa solo il bordo esterno come area e l'interno come ostacolo
    area_coords_3_b2_corrected = outer_boundary_b2
    obstacle_coords_3_b2_corrected = [
        inner_hole_b2, # Il buco interno diventa un ostacolo
        [(2, 48), (5, 48), (5, 49), (2, 49), (2, 48)],    # Ostacolo sul bordo
        [(45, 2), (48, 2), (48, 5), (45, 5), (45, 2)]    # Ostacolo sul bordo
    ]
    work_width_3_b2 = 2.5
    num_headlands_3_b2 = 2

    # Usa le coordinate corrette per l'inizializzazione
    calculator3_b2 = HeadlandCalculator(area_coords_3_b2_corrected, obstacle_coords_3_b2_corrected)
    headland_trajectories_3_b2, inner_area_coords_3_b2 = calculator3_b2.calculate_headlands(work_width_3_b2, num_headlands_3_b2)

    fig3_b2, ax3_b2 = plt.subplots(figsize=(10, 10))
    # Quando tracci l'area originale, usa il Polygon che ha anche il buco per la visualizzazione
    plot_polygon(ax3_b2, Polygon(outer_boundary_b2, [inner_hole_b2]), facecolor='blue', edgecolor='black', alpha=0.3, label='Area Originale')
    for obstacle in obstacle_coords_3_b2_corrected: # Traccia tutti gli ostacoli, incluso il buco
        plot_polygon(ax3_b2, Polygon(obstacle), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacoli')
    for i, trajectory in enumerate(headland_trajectories_3_b2):
        plot_segments(ax3_b2, [trajectory], color=f'C{i+1}', linewidth=2, linestyle='--', label=f'Capezzagna {i+1}')
    if inner_area_coords_3_b2:
        plot_polygon(ax3_b2, Polygon(inner_area_coords_3_b2), facecolor='green', edgecolor='green', alpha=0.3, label='Area Interna Finale')
    ax3_b2.set_aspect('equal', adjustable='box')
    ax3_b2.set_title(f"Campo a Ciambella con Ostacoli al Bordo - {len(headland_trajectories_3_b2)} Capezzagne")
    plt.xlabel("Coordinate X")
    plt.ylabel("Coordinate Y")
    plt.grid(True)
    plt.legend()
    plt.show()