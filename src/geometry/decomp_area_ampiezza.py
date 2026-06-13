from shapely.geometry import Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, split
import matplotlib.pyplot as plt
import numpy as np

class AreaCoverageProblem:
    def __init__(self, area_coordinates, obstacle_coordinates, cell_width=None):
        """
        Inizializza il problema di copertura dell'area.

        Args:
            area_coordinates (list): Lista di tuple (x, y) che definiscono il poligono dell'area.
            obstacle_coordinates (list): Lista di liste di tuple (x, y) per ogni ostacolo.
            cell_width (float, optional): Ampiezza desiderata per le sottocelle. Defaults to None.
        """
        self.area = Polygon(area_coordinates)
        self.obstacles = [Polygon(coords) for coords in obstacle_coordinates]
        self.cells = []
        self.cell_width = cell_width  # Conserva l'ampiezza desiderata
        self.generated_splitters = [] # Nuovo attributo per memorizzare le linee di divisione generate

    def decomponi_area(self, metodo="ottimizzata"):
        """
        Decompone l'area in sottocelle utilizzando il metodo specificato.
        Ora supporta "ottimizzata" oltre a "trapezoidale".
        """
        # Resetta le linee di divisione generate ad ogni nuova decomposizione
        self.generated_splitters = [] 
        if metodo == "trapezoidale":
            self.cells = self._decomposizione_trapezoidale()
        elif metodo == "ottimizzata":
            self.cells = self._decomposizione_ottimizzata()  # Usa il nuovo metodo
        else:
            raise ValueError("Metodo di decomposizione non supportato.")

    def _decomposizione_trapezoidale(self):
        """
        Esegue una decomposizione trapezoidale dell'area utilizzando shapely.ops.split.
        Questa implementazione è più robusta rispetto a un approccio manuale sweep-line.
        """
        cells = []

        # 1. Calcola l'area effettiva (area - ostacoli)
        effective_area = self.area
        for obstacle in self.obstacles:
            effective_area = effective_area.difference(obstacle)

        # Se l'area effettiva è vuota o non è un poligono valido, non ci sono celle
        if effective_area.is_empty or not effective_area.is_valid:
            print("L'area effettiva è vuota o non valida dopo aver rimosso gli ostacoli.")
            return []

        # 2. Estrai tutte le coordinate X uniche dai bordi dell'area effettiva
        # Questo include i vertici dell'area esterna e gli eventuali buchi/ostacoli
        all_x_coords = set()
        if effective_area.geom_type == 'Polygon':
            all_x_coords.update(p[0] for p in effective_area.exterior.coords)
            for interior in effective_area.interiors:
                all_x_coords.update(p[0] for p in interior.coords)
        elif effective_area.geom_type == 'MultiPolygon':
            for poly in effective_area.geoms:
                all_x_coords.update(p[0] for p in poly.exterior.coords)
                for interior in poly.interiors:
                    all_x_coords.update(p[0] for p in interior.coords)
        
        # Ordina le coordinate X
        sorted_x_coords = sorted(list(all_x_coords))

        # 3. Genera le linee verticali per la divisione
        minx, miny, maxx, maxy = self.area.bounds
        # Estendi le linee verticali leggermente oltre i limiti Y dell'area
        y_extension = (maxy - miny) * 0.1 # Un 10% di estensione
        vertical_lines = []
        for x_coord in sorted_x_coords:
            # Crea una linea verticale che attraversa l'intera estensione Y dell'area
            line = LineString([(x_coord, miny - y_extension), (x_coord, maxy + y_extension)])
            vertical_lines.append(line)

        # Unisci tutte le linee verticali in un MultiLineString per lo split
        if not vertical_lines:
            # Se non ci sono linee verticali (es. poligono a striscia verticale), l'area stessa è la cella
            if effective_area.geom_type == 'Polygon':
                return [effective_area]
            elif effective_area.geom_type == 'MultiPolygon':
                return list(effective_area.geoms)
            return [] # Nessuna cella se l'area è vuota o non gestita

        splitter = unary_union(vertical_lines)

        # 4. Esegui lo split dell'area effettiva con le linee verticali
        # Il risultato può essere un GeometryCollection o un MultiPolygon
        decomposed_geometry = split(effective_area, splitter)
        
        # 5. Estrai i poligoni validi dalla geometria decomposta
        for geom in decomposed_geometry.geoms:
            if geom.geom_type == 'Polygon' and geom.is_valid and not geom.is_empty:
                # Assicurati che la cella sia contenuta nell'area originale e non si sovrapponga agli ostacoli
                # Sottrai gli ostacoli dalla cella, anche se l'abbiamo già fatto per effective_area,
                # per gestire casi limite o imprecisioni flottanti.
                clean_cell = geom
                for obstacle in self.obstacles:
                    clean_cell = clean_cell.difference(obstacle)
                
                if clean_cell.geom_type == 'Polygon' and clean_cell.is_valid and not clean_cell.is_empty:
                    cells.append(clean_cell)
                elif clean_cell.geom_type == 'MultiPolygon':
                    # Se la differenza crea un MultiPolygon, aggiungi ogni parte come cella separata
                    for part in clean_cell.geoms:
                        if part.geom_type == 'Polygon' and part.is_valid and not part.is_empty:
                            cells.append(part)

        return cells

    def _decomposizione_ottimizzata(self):
        """
        Decompone l'area in sottocelle con ampiezza prestabilita, allineate al lato più lungo.
        """

        if self.cell_width is None:
            # Se cell_width non è fornito, torna alla decomposizione trapezoidale standard
            print("cell_width non fornito, usando la decomposizione trapezoidale standard.")
            return self._decomposizione_trapezoidale()

        cells = []
        effective_area = self.area
        for obstacle in self.obstacles:
            effective_area = effective_area.difference(obstacle)

        if effective_area.is_empty or not effective_area.is_valid:
            print("L'area effettiva è vuota o non valida dopo aver rimosso gli ostacoli.")
            return []

        minx, miny, maxx, maxy = self.area.bounds
        width = maxx - minx
        height = maxy - miny

        # Determina l'orientamento delle divisioni (lungo il lato più lungo)
        if width > height:
            num_divisions = int(np.ceil(width / self.cell_width))
            is_vertical = True  # Divisioni verticali
            start_coord = minx
            end_coord = maxx
        else:
            num_divisions = int(np.ceil(height / self.cell_width))
            is_vertical = False # Divisioni orizzontali
            start_coord = miny
            end_coord = maxy

        # Genera le linee di divisione
        splitters = []
        # Estendi le linee leggermente oltre i limiti dell'area per garantire un taglio completo
        x_extension = (maxx - minx) * 0.1
        y_extension = (maxy - miny) * 0.1

        for i in range(1, num_divisions):  # Inizia da 1 per evitare la linea sul bordo iniziale
            if is_vertical:
                x = start_coord + i * self.cell_width
                # Assicurati che la linea non superi il bordo finale
                if x < end_coord:
                    line = LineString([(x, miny - y_extension), (x, maxy + y_extension)])
                    splitters.append(line)
                    self.generated_splitters.append(line) # Memorizza la linea generata
            else:
                y = start_coord + i * self.cell_width
                # Assicurati che la linea non superi il bordo finale
                if y < end_coord:
                    line = LineString([(minx - x_extension, y), (maxx + x_extension, y)])
                    splitters.append(line)
                    self.generated_splitters.append(line) # Memorizza la linea generata

        # Unisci le linee di divisione per lo split
        if splitters:
            splitter = unary_union(splitters)
            decomposed_geometry = split(effective_area, splitter)

            # Estrai i poligoni validi dalla geometria decomposta
            for geom in decomposed_geometry.geoms:
                # Controlla se il risultato dello split è un Polygon e se è valido e non vuoto
                if geom.geom_type == 'Polygon' and geom.is_valid and not geom.is_empty:
                    clean_cell = geom
                    # Rimuovi gli ostacoli dalla cella risultante
                    for obstacle in self.obstacles:
                        clean_cell = clean_cell.difference(obstacle)
                    
                    # Aggiungi la cella pulita se è ancora un poligono valido e non vuoto
                    if clean_cell.geom_type == 'Polygon' and clean_cell.is_valid and not clean_cell.is_empty:
                        cells.append(clean_cell)
                    elif clean_cell.geom_type == 'MultiPolygon':
                        # Se la differenza crea un MultiPolygon, aggiungi ogni parte come cella separata
                        for part in clean_cell.geoms:
                            if part.geom_type == 'Polygon' and part.is_valid and not part.is_empty:
                                cells.append(part)
                elif geom.geom_type == 'MultiPolygon':
                    # Se lo split produce direttamente un MultiPolygon, itera sulle sue parti
                    for part in geom.geoms:
                        if part.geom_type == 'Polygon' and part.is_valid and not part.is_empty:
                            clean_cell = part
                            for obstacle in self.obstacles:
                                clean_cell = clean_cell.difference(obstacle)
                            if clean_cell.geom_type == 'Polygon' and clean_cell.is_valid and not clean_cell.is_empty:
                                cells.append(clean_cell)
                            elif clean_cell.geom_type == 'MultiPolygon':
                                for sub_part in clean_cell.geoms:
                                    if sub_part.geom_type == 'Polygon' and sub_part.is_valid and not sub_part.is_empty:
                                        cells.append(sub_part)
        else:
            # Se non ci sono linee di divisione (es. l'area è più piccola di cell_width),
            # l'area effettiva stessa è la cella o le sue componenti
            if effective_area.geom_type == 'Polygon':
                cells.append(effective_area)
            elif effective_area.geom_type == 'MultiPolygon':
                cells.extend(effective_area.geoms)

        return cells

# --- Funzioni di Visualizzazione ---
def plot_polygon(ax, polygon, facecolor='blue', edgecolor='black', alpha=0.5):
    """Disegna un poligono sul grafico con un colore di riempimento e un bordo."""
    if polygon.geom_type == 'Polygon':
        x, y = polygon.exterior.xy
        ax.fill(x, y, facecolor=facecolor, alpha=alpha)
        ax.plot(x, y, color=edgecolor, linewidth=1) # Disegna il contorno
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.fill(x, y, facecolor='white', alpha=1) # I buchi sono bianchi
            ax.plot(x, y, color=edgecolor, linewidth=1) # Disegna il contorno del buco
    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha) # Ricorsivamente disegna ogni poligono

def plot_segments(ax, segments, color='black', linestyle='-', linewidth=1):
    """Disegna una lista di segmenti di linea sul grafico."""
    for segment in segments:
        x, y = segment.xy
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)

def plot_points(ax, points, color='red', marker='o', markersize=None, label=None, linestyle='None'):
    """Disegna una lista di punti sul grafico."""
    # points deve essere una lista di tuple (x, y), ad esempio [(x1, y1), (x2, y2)]
    if not points: # Gestisce il caso di lista vuota
        return
    x, y = zip(*points)
    # Passa markersize ad ax.plot se fornito
    if markersize is not None:
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, markersize=markersize, label=label)
    else:
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, label=label)


if __name__ == '__main__':
    # Esempio 1: Poligono semplice con ostacolo e ampiezza cella
    print("--- Esempio 1: Poligono semplice con ostacolo e ampiezza cella ---")
    area_coords_1 = [(0, 0), (10, 0), (10, 10), (0, 10), (0,0)]
    obstacle_coords_1 = [[(3, 3), (7, 3), (7, 7), (3, 7), (3,3)]]
    cell_width_1 = 3  # Ampiezza desiderata delle celle (editabile qui)

    problem1 = AreaCoverageProblem(area_coords_1, obstacle_coords_1, cell_width_1)
    problem1.decomponi_area(metodo="ottimizzata")  # Usa il metodo ottimizzato

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax1, problem1.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem1.obstacles:
        plot_polygon(ax1, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem1.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem1.cells):
            if len(problem1.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem1.cells) - 1))
            plot_polygon(ax1, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem1.generated_splitters:
        plot_segments(ax1, problem1.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(f"Decomposizione Ottimizzata (Esempio 1) - {len(problem1.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem1.cells)}")

    # Esempio 2: Poligono concavo senza ostacoli e ampiezza cella
    print("--- Esempio 2: Poligono concavo senza ostacoli e ampiezza cella ---")
    area_coords_2 = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10), (0,0)]
    obstacle_coords_2 = []
    cell_width_2 = 2  # Ampiezza desiderata delle celle (editabile qui)

    problem2 = AreaCoverageProblem(area_coords_2, obstacle_coords_2, cell_width_2)
    problem2.decomponi_area(metodo="ottimizzata")

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax2, problem2.area, facecolor='blue', edgecolor='black', alpha=0.3)
    if len(problem2.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem2.cells):
            if len(problem2.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem2.cells) - 1))
            plot_polygon(ax2, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem2.generated_splitters:
        plot_segments(ax2, problem2.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(f"Decomposizione Ottimizzata (Esempio 2) - {len(problem2.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem2.cells)}")

    # Esempio 3: Poligono con un "buco" e ampiezza cella
    print("--- Esempio 3: Poligono con un 'buco' e ampiezza cella ---")
    area_coords_3 = [(0, 0), (10, 0), (10, 10), (0, 10), (0,0)]
    obstacle_coords_3 = [[(2, 2), (8, 2), (8, 8), (2, 8), (2,2)]] # Un buco grande
    cell_width_3 = 4  # Ampiezza desiderata delle celle (editabile qui)

    problem3 = AreaCoverageProblem(area_coords_3, obstacle_coords_3, cell_width_3)
    problem3.decomponi_area(metodo="ottimizzata")

    fig3, ax3 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax3, problem3.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem3.obstacles:
        plot_polygon(ax3, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem3.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem3.cells):
            if len(problem3.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem3.cells) - 1))
            plot_polygon(ax3, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem3.generated_splitters:
        plot_segments(ax3, problem3.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title(f"Decomposizione Ottimizzata (Esempio 3) - {len(problem3.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem3.cells)}")

    # --- Esempio 4: Poligono a forma di stella con ostacoli multipli e ampiezza cella ---
    print("--- Esempio 4: Poligono a forma di stella con ostacoli multipli e ampiezza cella ---")
    area_coords_4 = [
        (5, 0), (6, 4), (10, 5), (6, 6), (5, 10), (4, 6), (0, 5), (4, 4), (5, 0)
    ]
    obstacle_coords_4 = [
        [ (2, 2), (3, 2), (3, 3), (2, 3), (2,2) ], # Ostacolo 1
        [ (7, 7), (8, 7), (8, 8), (7, 8), (7,7) ], # Ostacolo 2
        [ (4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5), (4.5,4.5) ] # Ostacolo 3 al centro
    ]
    cell_width_4 = 2.5 # Ampiezza desiderata delle celle (editabile qui)

    problem4 = AreaCoverageProblem(area_coords_4, obstacle_coords_4, cell_width_4)
    problem4.decomponi_area(metodo="ottimizzata")

    fig4, ax4 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax4, problem4.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem4.obstacles:
        plot_polygon(ax4, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem4.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem4.cells):
            if len(problem4.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem4.cells) - 1))
            plot_polygon(ax4, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem4.generated_splitters:
        plot_segments(ax4, problem4.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title(f"Decomposizione Ottimizzata (Esempio 4) - {len(problem4.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem4.cells)}")

    # --- Esempio 5: Poligono irregolare complesso con ostacoli sparsi e ampiezza cella ---
    print("--- Esempio 5: Poligono irregolare complesso con ostacoli sparsi e ampiezza cella ---")
    area_coords_5 = [
        (0,0), (12,0), (10,3), (12,6), (8,8), (6,10), (4,9), (2,7), (0,5), (0,0)
    ]
    obstacle_coords_5 = [
        [(1,1), (2,1), (2,2), (1,2), (1,1)],
        [(4,2), (5,2), (5,3), (4,3), (4,2)],
        [(7,1), (8,1), (8,2), (7,2), (7,1)],
        [(3,6), (4,6), (4,7), (3,7), (3,6)],
        [(9,4), (10,4), (10,5), (9,5), (9,4)]
    ]
    cell_width_5 = 3.5 # Ampiezza desiderata delle celle (editabile qui)

    problem5 = AreaCoverageProblem(area_coords_5, obstacle_coords_5, cell_width_5)
    problem5.decomponi_area(metodo="ottimizzata")

    fig5, ax5 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax5, problem5.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem5.obstacles:
        plot_polygon(ax5, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem5.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem5.cells):
            if len(problem5.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem5.cells) - 1))
            plot_polygon(ax5, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem5.generated_splitters:
        plot_segments(ax5, problem5.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax5.set_aspect('equal', adjustable='box')
    ax5.set_title(f"Decomposizione Ottimizzata (Esempio 5) - {len(problem5.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem5.cells)}")

    # --- Esempio 6: Area a L con ostacoli complessi e ampiezza cella ---
    print("--- Esempio 6: Area a L con ostacoli complessi e ampiezza cella ---")
    area_coords_6 = [
        (0,0), (10,0), (10,5), (5,5), (5,10), (0,10), (0,0)
    ]
    obstacle_coords_6 = [
        [(2,2), (3,2), (3,3), (2,3), (2,2)],
        [(7,1), (8,1), (8,2), (7,2), (7,1)],
        [(6,7), (7,7), (7,8), (6,8), (6,7)],
        [(2,7), (3,7), (3,8), (2,8), (2,7)]
    ]
    cell_width_6 = 2.0 # Ampiezza desiderata delle celle (editabile qui)

    problem6 = AreaCoverageProblem(area_coords_6, obstacle_coords_6, cell_width_6)
    problem6.decomponi_area(metodo="ottimizzata")

    fig6, ax6 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax6, problem6.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem6.obstacles:
        plot_polygon(ax6, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem6.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem6.cells):
            if len(problem6.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem6.cells) - 1))
            plot_polygon(ax6, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem6.generated_splitters:
        plot_segments(ax6, problem6.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax6.set_aspect('equal', adjustable='box')
    ax6.set_title(f"Decomposizione Ottimizzata (Esempio 6) - {len(problem6.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem6.cells)}")

    # Esempio 7: Area a C con ostacoli allungati e ampiezza cella ---
    print("--- Esempio 7: Area a C con ostacoli allungati e ampiezza cella ---")
    area_coords_7 = [
        (0,0), (12,0), (12,10), (10,10), (10,2), (2,2), (2,10), (0,10), (0,0)
    ]
    obstacle_coords_7 = [
        [(3,4), (4,4), (4,8), (3,8), (3,4)], # Ostacolo verticale
        [(6,1), (8,1), (8,2), (6,2), (6,1)], # Ostacolo orizzontale basso
        [(6,8), (8,8), (8,9), (6,9), (6,8)]  # Ostacolo orizzontale alto
    ]
    cell_width_7 = 3.0 # Ampiezza desiderata delle celle (editabile qui)

    problem7 = AreaCoverageProblem(area_coords_7, obstacle_coords_7, cell_width_7)
    problem7.decomponi_area(metodo="ottimizzata")

    fig7, ax7 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax7, problem7.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem7.obstacles:
        plot_polygon(ax7, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem7.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem7.cells):
            if len(problem7.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem7.cells) - 1))
            plot_polygon(ax7, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem7.generated_splitters:
        plot_segments(ax7, problem7.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax7.set_aspect('equal', adjustable='box')
    ax7.set_title(f"Decomposizione Ottimizzata (Esempio 7) - {len(problem7.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem7.cells)}")

    # --- Esempio 8: Area a forma di "U" con ostacoli circolari e triangolari e ampiezza cella ---
    print("--- Esempio 8: Area a forma di 'U' con ostacoli circolari e triangolari e ampiezza cella ---")
    area_coords_8 = [
        (0,0), (10,0), (10,10), (8,10), (8,2), (2,2), (2,10), (0,10), (0,0)
    ]
    # Ostacolo circolare (approssimato con un poligono a molti lati)
    num_segments = 20
    center_x_circle, center_y_circle = 5, 6
    radius_circle = 1.5
    circle_obstacle_coords = []
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = center_x_circle + radius_circle * np.cos(angle)
        y = center_y_circle + radius_circle * np.sin(angle)
        circle_obstacle_coords.append((x, y))
    circle_obstacle_coords.append(circle_obstacle_coords[0]) # Chiudi il poligono

    obstacle_coords_8 = [
        circle_obstacle_coords, # Ostacolo circolare
        [(1,4), (2,6), (3,4), (1,4)], # Ostacolo triangolare 1
        [(7,4), (9,4), (8,6), (7,4)]  # Ostacolo triangolare 2
    ]
    cell_width_8 = 2.5 # Ampiezza desiderata delle celle (editabile qui)

    problem8 = AreaCoverageProblem(area_coords_8, obstacle_coords_8, cell_width_8)
    problem8.decomponi_area(metodo="ottimizzata")

    fig8, ax8 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax8, problem8.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem8.obstacles:
        plot_polygon(ax8, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem8.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem8.cells):
            if len(problem8.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem8.cells) - 1))
            plot_polygon(ax8, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem8.generated_splitters:
        plot_segments(ax8, problem8.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax8.set_aspect('equal', adjustable='box')
    ax8.set_title(f"Decomposizione Ottimizzata (Esempio 8) - {len(problem8.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem8.cells)}")

    # --- Esempio 9: Area complessa con ostacoli a forma di L e irregolari e ampiezza cella ---
    print("--- Esempio 9: Area complessa con ostacoli a forma di L e irregolari e ampiezza cella ---")
    area_coords_9 = [
        (0,0), (15,0), (15,12), (10,12), (10,4), (5,4), (5,12), (0,12), (0,0)
    ]
    obstacle_coords_9 = [
        # Ostacolo a L
        [(3,2), (6,2), (6,3), (4,3), (4,5), (3,5), (3,2)],
        # Ostacolo a forma di diamante
        [(8,7), (9,8), (8,9), (7,8), (8,7)],
        # Ostacolo irregolare
        [(12,1), (13,2), (12.5,4), (11.5,3), (12,1)]
    ]
    cell_width_9 = 4.0 # Ampiezza desiderata delle celle (editabile qui)

    problem9 = AreaCoverageProblem(area_coords_9, obstacle_coords_9, cell_width_9)
    problem9.decomponi_area(metodo="ottimizzata")

    fig9, ax9 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax9, problem9.area, facecolor='blue', edgecolor='black', alpha=0.3)
    for obstacle in problem9.obstacles:
        plot_polygon(ax9, obstacle, facecolor='black', edgecolor='black', alpha=1.0)
    
    if len(problem9.cells) > 0:
        cmap = plt.colormaps.get_cmap('viridis')
        for i, cell in enumerate(problem9.cells):
            if len(problem9.cells) == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (len(problem9.cells) - 1))
            plot_polygon(ax9, cell, facecolor=color, edgecolor='black', alpha=0.8)
    
    # Disegna le linee di divisione interne per il metodo ottimizzato
    if problem9.generated_splitters:
        plot_segments(ax9, problem9.generated_splitters, color='red', linestyle='--', linewidth=1.5)

    ax9.set_aspect('equal', adjustable='box')
    ax9.set_title(f"Decomposizione Ottimizzata (Esempio 9) - {len(problem9.cells)} Celle")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()
    print(f"Numero di celle generate: {len(problem9.cells)}")
