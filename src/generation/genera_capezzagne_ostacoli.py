from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np

def calcola_capezzagne(obstacle_coordinates, ampiezza_lavoro, num_concentric_layers=1, buffer_resolution=16):
    """
    Calcola le capezzagne (buffer) attorno a una lista di ostacoli e le traiettorie interne,
    generando N segmenti concentrici e raggruppando gli ostacoli vicini.

    Args:
        obstacle_coordinates (list): Lista di liste di tuple (x, y) che definiscono i poligoni degli ostacoli.
                                    Ogni elemento della lista esterna è un ostacolo.
        ampiezza_lavoro (float): La larghezza di un singolo strato concentrico di capezzagna.
                                 La traiettoria per questo strato sarà posizionata al centro.
        num_concentric_layers (int, optional): Il numero di segmenti concentrici di capezzagna da generare. Default è 1.
                                                Se non valido (es. non intero, negativo o < 2), verrà forzato a 2.
        buffer_resolution (int, optional): Il numero di segmenti da usare per approssimare
                                            curve quando si crea il buffer. Più alto è il numero,
                                            più liscia è l'approssimazione. Default è 16.

    Returns:
        tuple: Una tupla contenente due liste di liste:
            - all_concentric_capezzagne_layers (list of lists of Polygon): Ogni lista interna contiene
              i poligoni di capezzagna per un determinato strato concentrico, uniti per gruppi di ostacoli.
            - all_concentric_trajectories_layers (list of lists of LineString): Ogni lista interna contiene
              le traiettorie per un determinato strato concentrico, posizionate al centro di ogni segmento
              e uniti per gruppi di ostacoli.
    """
    all_concentric_capezzagne_layers = []
    all_concentric_trajectories_layers = []

    # --- Controllo per num_concentric_layers ---
    num_headlands_effective = num_concentric_layers
    # Se non è un intero, o se è minore di 1 
    if not isinstance(num_headlands_effective, int) or num_headlands_effective < 0:
        print(f"Attenzione: 'num_concentric_layers' non valido ({num_concentric_layers}). Forzo a 2.")
        num_headlands_effective =  1
    # --- Fine controllo ---

    # Se non ci sono ostacoli, non ci sono capezzagne
    if not obstacle_coordinates:
        return [], []

    # 1. Create Shapely Polygon objects for all original obstacles
    original_obstacles_polygons = [Polygon(coords) for coords in obstacle_coordinates]

    # Calcola la larghezza totale che sarà coperta da tutti gli strati di capezzagna.
    # Questa è necessaria per il buffer iniziale di raggruppamento.
    total_buffer_for_grouping = ampiezza_lavoro * num_headlands_effective

    # 2. Raggruppa gli ostacoli in base alla prossimità
    initial_proximity_buffers = [p.buffer(total_buffer_for_grouping, resolution=buffer_resolution) for p in original_obstacles_polygons]
    
    # Unisci tutti i buffer iniziali per trovare i componenti connessi
    unioned_proximity_buffers = unary_union(initial_proximity_buffers)

    # Estrai i singoli componenti connessi (gruppi)
    if unioned_proximity_buffers.geom_type == 'Polygon':
        connected_buffer_components = [unioned_proximity_buffers]
    elif unioned_proximity_buffers.geom_type == 'MultiPolygon':
        connected_buffer_components = list(unioned_proximity_buffers.geoms)
    elif unioned_proximity_buffers.is_empty:
        return [], [] # Nessun ostacolo, nessuna capezzagna
    else:
        print(f"Warning: Unexpected geometry type {unioned_proximity_buffers.geom_type} for unioned_proximity_buffers. Returning empty.")
        return [], []


    # Per ogni componente connesso, determina l'ostacolo "effettivo"
    # Questa è l'unione degli ostacoli ORIGINALI che rientrano in questo componente
    effective_obstacles_for_groups = []
    for component_buffer in connected_buffer_components:
        obstacles_in_this_group = []
        for original_obs_poly in original_obstacles_polygons:
            if component_buffer.intersects(original_obs_poly):
                obstacles_in_this_group.append(original_obs_poly)
        
        if obstacles_in_this_group:
            effective_obstacles_for_groups.append(unary_union(obstacles_in_this_group))


    # 3. Genera strati concentrici e traiettorie per ogni gruppo di ostacoli effettivo
    for i in range(1, num_headlands_effective + 1): # Itera per il numero effettivo di strati
        current_layer_capezzagne = []
        current_layer_trajectories = []

        # Calcola i raggi esterno e interno per lo strato corrente
        # ampiezza_lavoro è ora la larghezza di UN singolo strato
        current_outer_radius = i * ampiezza_lavoro
        current_inner_radius = (i - 1) * ampiezza_lavoro
        # La traiettoria è al centro del segmento
        current_trajectory_radius = current_inner_radius + (ampiezza_lavoro / 2) 

        for effective_obs in effective_obstacles_for_groups:
            # Genera il segmento di capezzagna (anello) per questo ostacolo effettivo
            outer_buffer_for_group = effective_obs.buffer(current_outer_radius, resolution=buffer_resolution)
            
            if i == 1: # Per il primo strato, il bordo interno è l'ostacolo stesso
                inner_buffer_for_group = effective_obs
            else: # Per gli strati successivi, il bordo interno è un buffer dell'ostacolo
                inner_buffer_for_group = effective_obs.buffer(current_inner_radius, resolution=buffer_resolution)

            segment_poly = outer_buffer_for_group.difference(inner_buffer_for_group)

            if segment_poly.geom_type == 'Polygon':
                current_layer_capezzagne.append(segment_poly)
            elif segment_poly.geom_type == 'MultiPolygon':
                current_layer_capezzagne.extend(list(segment_poly.geoms))
            elif not segment_poly.is_empty:
                print(f"Warning: Unexpected geometry type {segment_poly.geom_type} for capezzagna segment {i} of effective obstacle. Skipping.")

            # Genera la traiettoria per questo ostacolo effettivo per questo strato
            trajectory_line_candidate = effective_obs.buffer(current_trajectory_radius, resolution=buffer_resolution)
            
            if trajectory_line_candidate.geom_type == 'LineString':
                current_layer_trajectories.append(trajectory_line_candidate)
            elif trajectory_line_candidate.geom_type == 'MultiLineString':
                current_layer_trajectories.extend(list(trajectory_line_candidate.geoms))
            elif trajectory_line_candidate.geom_type == 'Polygon': # Può accadere se il buffer è molto grande e riempie
                current_layer_trajectories.append(trajectory_line_candidate.exterior)
            elif trajectory_line_candidate.geom_type == 'MultiPolygon':
                current_layer_trajectories.extend([p.exterior for p in trajectory_line_candidate.geoms if p.geom_type == 'Polygon'])
            elif trajectory_line_candidate.geom_type == 'GeometryCollection':
                for geom in trajectory_line_candidate.geoms:
                    if geom.geom_type == 'LineString':
                        current_layer_trajectories.append(geom)
                    elif geom.geom_type == 'Polygon':
                        current_layer_trajectories.append(geom.exterior)
            elif not trajectory_line_candidate.is_empty:
                print(f"Warning: Unexpected geometry type {trajectory_line_candidate.geom_type} for trajectory buffer for effective obstacle. Skipping.")
            
        all_concentric_capezzagne_layers.append(current_layer_capezzagne)
        all_concentric_trajectories_layers.append(current_layer_trajectories)

    return all_concentric_capezzagne_layers, all_concentric_trajectories_layers

def genera_ostacoli_ingranditi(obstacle_coordinates, all_capezzagne_layers):
    """
    Genera una lista di ostacoli "ingranditi" includendo l'area delle capezzagne.

    Args:
        obstacle_coordinates (list): Lista di liste di tuple (x, y) che definiscono i poligoni degli ostacoli originali.
        all_capezzagne_layers (list of lists of Polygon): Lista di liste di poligoni di capezzagna, come restituito da calcola_capezzagne().

    Returns:
        list: Una lista di liste di tuple (x, y) che definiscono i poligoni degli ostacoli "ingranditi".
    """

    # Converti gli ostacoli originali in oggetti Polygon di Shapely
    original_obstacles_polygons = [Polygon(coords) for coords in obstacle_coordinates]

    # Se non ci sono capezzagne, restituisci gli ostacoli originali
    if not all_capezzagne_layers:
        return [list(p.exterior.coords) for p in original_obstacles_polygons]

    final_enlarged_obstacles = []

    # Unisci tutti i poligoni della capezzagna più esterna in un unico MultiPolygon o Polygon.
    if all_capezzagne_layers:
        outermost_capezzagna_geoms = all_capezzagne_layers[-1]
        if outermost_capezzagna_geoms:
            unioned_outermost_capezzagna = unary_union(outermost_capezzagna_geoms)
        else:
            unioned_outermost_capezzagna = GeometryCollection()
    else:
        unioned_outermost_capezzagna = GeometryCollection()

    if unioned_outermost_capezzagna.is_empty:
        # Se non c'è una capezzagna più esterna (es. nessun ostacolo o ampiezza_lavoro è 0)
        # restituisci gli ostacoli originali.
        return [list(p.exterior.coords) for p in original_obstacles_polygons]


    # Iteriamo sui componenti della capezzagna unita più esterna
    # Ogni componente corrisponde a un gruppo di ostacoli effettivo.
    if unioned_outermost_capezzagna.geom_type == 'Polygon':
        components_to_process = [unioned_outermost_capezzagna]
    elif unioned_outermost_capezzagna.geom_type == 'MultiPolygon':
        components_to_process = list(unioned_outermost_capezzagna.geoms)
    else:
        # Potrebbe essere GeometryCollection, LineString (se degenerato), ecc.
        # In tal caso, non c'è un'area ingrandita significativa oltre l'originale.
        print(f"Warning: Tipo di geometria inatteso per la capezzagna più esterna unita: {unioned_outermost_capezzagna.geom_type}. Restituisco ostacoli originali.")
        return [list(p.exterior.coords) for p in original_obstacles_polygons]


    processed_original_obstacles_indices = set() # Per tenere traccia degli ostacoli originali già inclusi in un gruppo ingrandito

    for component_outer_capezzagna in components_to_process:
        # Inizializza con l'ostacolo più grande/rappresentativo di questo gruppo, o un GeometryCollection
        group_obstacles_union = GeometryCollection()
        
        # Trova gli ostacoli originali che intersecano questo componente di capezzagna
        current_group_original_obstacles = []
        for i, original_obs_poly in enumerate(original_obstacles_polygons):
            if component_outer_capezzagna.intersects(original_obs_poly) and i not in processed_original_obstacles_indices:
                current_group_original_obstacles.append(original_obs_poly)
                processed_original_obstacles_indices.add(i)

        if current_group_original_obstacles:
            group_obstacles_union = unary_union(current_group_original_obstacles)
        else:
            # Questo caso può succedere se un componente della capezzagna esterna non interseca
            # direttamente un ostacolo originale (es. se la capezzagna è molto grande e si estende).
            # In questo caso, consideriamo la capezzagna stessa come l'ostacolo ingrandito.
            if component_outer_capezzagna.geom_type == 'Polygon':
                final_enlarged_obstacles.append(list(component_outer_capezzagna.exterior.coords))
            elif component_outer_capezzagna.geom_type == 'MultiPolygon':
                 for poly in component_outer_capezzagna.geoms:
                    if poly.is_valid and poly.geom_type == 'Polygon':
                        final_enlarged_obstacles.append(list(poly.exterior.coords))
            continue # Passa al prossimo componente se non ci sono ostacoli originali collegati

        # Unisci gli ostacoli originali del gruppo con il loro componente di capezzagna più esterna
        enlarged_shape = unary_union([group_obstacles_union, component_outer_capezzagna])

        # Assicurati che il risultato sia un poligono o MultiPolygon valido
        if enlarged_shape.is_valid and enlarged_shape.geom_type == 'Polygon':
            final_enlarged_obstacles.append(list(enlarged_shape.exterior.coords))
        elif enlarged_shape.geom_type == 'MultiPolygon':
            for poly in enlarged_shape.geoms:
                if poly.is_valid and poly.geom_type == 'Polygon':
                    final_enlarged_obstacles.append(list(poly.exterior.coords))
        else:
            print(f"Warning: Forma ingrandita finale non è un poligono valido o MultiPolygon. Skipping. Tipo: {enlarged_shape.geom_type}")
            # Fallback: se non valido, aggiungi il contorno dell'unione degli ostacoli originali del gruppo
            if group_obstacles_union.is_valid and group_obstacles_union.geom_type == 'Polygon':
                final_enlarged_obstacles.append(list(group_obstacles_union.exterior.coords))
            elif group_obstacles_union.geom_type == 'MultiPolygon':
                for poly in group_obstacles_union.geoms:
                    if poly.is_valid and poly.geom_type == 'Polygon':
                        final_enlarged_obstacles.append(list(poly.exterior.coords))


    # Aggiungi gli ostacoli originali che non sono stati intersecati da nessuna capezzagna
    for i, original_obs_poly in enumerate(original_obstacles_polygons):
        if i not in processed_original_obstacles_indices:
            if original_obs_poly.is_valid and original_obs_poly.geom_type == 'Polygon':
                final_enlarged_obstacles.append(list(original_obs_poly.exterior.coords))
            elif original_obs_poly.geom_type == 'MultiPolygon':
                for poly in original_obs_poly.geoms:
                    if poly.is_valid and poly.geom_type == 'Polygon':
                        final_enlarged_obstacles.append(list(poly.exterior.coords))

    return final_enlarged_obstacles


# --- Funzione di Visualizzazione (per testing) ---
def plot_polygon(ax, polygon, facecolor='blue', edgecolor='black', alpha=0.5, label=None, linewidth=1, linestyle='-'):
    """
    Disegna un poligono sul grafico con un colore di riempimento e un bordo.
    Aggiunti linewidth e linestyle per maggiore controllo sul bordo.
    """
    if polygon.geom_type == 'Polygon':
        x, y = polygon.exterior.xy
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, label=label)
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, linestyle=linestyle) # Passa linewidth e linestyle
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.fill(x, y, facecolor='white', alpha=1) # I buchi sono bianchi
            ax.plot(x, y, color=edgecolor, linewidth=linewidth, linestyle=linestyle) # Passa linewidth e linestyle anche per i buchi
    elif polygon.geom_type == 'MultiPolygon':
        for i, poly in enumerate(polygon.geoms):
            # Per MultiPolygon, solo il primo poligono avrà l'etichetta per la leggenda
            plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, 
                         label=label if i == 0 else None, 
                         linewidth=linewidth, linestyle=linestyle) # Passa anche a ricorsiva

def plot_lines(ax, lines, color='green', linestyle='--', linewidth=1.5, label=None):
    """Disegna una lista di LineString sul grafico."""
    for i, line in enumerate(lines):
        if line.geom_type == 'LineString':
            x, y = line.xy
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label if i == 0 else None)
        elif line.geom_type == 'MultiLineString':
            for single_line in line.geoms:
                x, y = single_line.xy
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label if i == 0 else None)
        elif line.geom_type == 'LinearRing': # Contorno di un poligono
            x, y = line.xy
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label if i == 0 else None)


if __name__ == '__main__':
    print("--- Esempi di Generazione Capezzagne e Traiettorie Interne ---")

    # Esempio 1: Ostacolo quadrato semplice con 3 strati di capezzagna
    # ampiezza_lavoro = 0.5 significa che ogni strato ha larghezza 0.5
    print("\nEsempio 1: Ostacolo quadrato semplice con 3 strati di capezzagna (larghezza singola 0.5)")
    obstacle_coords_1 = [[(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]]
    ampiezza_capezzagna_1 = 0.5 # Larghezza di OGNI singolo strato
    num_strati_1 = 3 # Numero di strati concentrici
    
    all_capezzagne_layers_1, all_trajectories_layers_1 = calcola_capezzagne(
        obstacle_coords_1, ampiezza_capezzagna_1, num_concentric_layers=num_strati_1
    )

    # Calcolo gli ostacoli ingranditi
    enlarged_obstacles_1 = genera_ostacoli_ingranditi(obstacle_coords_1, all_capezzagne_layers_1)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax1, Polygon(obstacle_coords_1[0]), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale')
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_strati_1)) # Colori per i diversi strati
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_1):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax1, capezzagna_poly, facecolor=colors[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_1):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax1, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    # Disegna gli ostacoli ingranditi
    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_1):
        plot_polygon(ax1, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(f"Capezzagne Concentriche, Traiettorie e Ostacoli Ingranditi (Esempio 1) - Larghezza Singolo Strato: {ampiezza_capezzagna_1}")
    ax1.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_1)}")
    print(f"Numero totale di strati di traiettorie: {len(all_trajectories_layers_1)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_1)}")


    # Esempio 2: Ostacoli multipli vicini che si sovrappongono con 2 strati di capezzagna
    # Ogni strato avrà larghezza 0.75
    print("\nEsempio 2: Ostacoli multipli vicini che si sovrappongono con 2 strati di capezzagna (UNIONE AUTOMATICA, larghezza singola 0.75)")
    obstacle_coords_2 = [
        [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)], # Quadrato piccolo 1
        [(2.5, 1), (3.5, 1), (3.5, 2), (2.5, 2), (2.5, 1)], # Quadrato piccolo 2 (vicino al primo)
        [(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)], # Quadrato piccolo 3
        [(6.5, 5), (7.5, 5), (7.5, 6), (6.5, 6), (6.5, 5)] # Quadrato piccolo 4 (vicino al terzo)
    ]
    ampiezza_capezzagna_2 = 0.75 # Larghezza di OGNI singolo strato
    num_strati_2 = 2 # Numero di strati concentrici

    # Ora calcola_capezzagne gestisce l'unione internamente
    all_capezzagne_layers_2, all_trajectories_layers_2 = calcola_capezzagne(
        obstacle_coords_2, ampiezza_capezzagna_2, num_concentric_layers=num_strati_2
    )

    enlarged_obstacles_2 = genera_ostacoli_ingranditi(obstacle_coords_2, all_capezzagne_layers_2)

    fig2, ax2 = plt.subplots(figsize=(8, 8)) # Un solo subplot per l'unione automatica
    
    for i, obstacle_coords in enumerate(obstacle_coords_2):
        plot_polygon(ax2, Polygon(obstacle_coords), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale' if i == 0 else None)
    
    colors_2 = plt.cm.Oranges(np.linspace(0.3, 0.9, num_strati_2))
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_2):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax2, capezzagna_poly, facecolor=colors_2[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Unita Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_2):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax2, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Unita Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_2):
        plot_polygon(ax2, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(f"Capezzagne Unite, Traiettorie e Ostacoli Ingranditi (Esempio 2) - Larghezza Singolo Strato: {ampiezza_capezzagna_2}")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Numero totale di strati di capezzagne unite: {len(all_capezzagne_layers_2)}")
    print(f"Numero totale di strati di traiettorie unite: {len(all_trajectories_layers_2)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_2)}")


    # Esempio 3: Ostacolo concavo (a forma di L) con 4 strati di capezzagna
    # Ogni strato avrà larghezza 0.4
    print("\nEsempio 3: Ostacolo concavo (a forma di L) con 4 strati di capezzagna (larghezza singola 0.4)")
    obstacle_coords_3 = [
        [(2,2), (6,2), (6,3), (4,3), (4,5), (2,5), (2,2)]
    ]
    ampiezza_capezzagna_3 = 0.4 # Larghezza di OGNI singolo strato
    num_strati_3 = 4
    all_capezzagne_layers_3, all_trajectories_layers_3 = calcola_capezzagne(
        obstacle_coords_3, ampiezza_capezzagna_3, num_concentric_layers=num_strati_3
    )

    enlarged_obstacles_3 = genera_ostacoli_ingranditi(obstacle_coords_3, all_capezzagne_layers_3)

    fig3, ax3 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax3, Polygon(obstacle_coords_3[0]), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale')
    
    colors_3 = plt.cm.magma(np.linspace(0.2, 0.8, num_strati_3))
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_3):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax3, capezzagna_poly, facecolor=colors_3[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_3):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax3, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_3):
        plot_polygon(ax3, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title(f"Capezzagne Concentriche, Traiettorie e Ostacoli Ingranditi (Esempio 3) - Larghezza Singolo Strato: {ampiezza_capezzagna_3}")
    ax3.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_3)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_3)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_3)}")


    # Esempio 4: Ostacoli multipli distanti con 2 strati
    # Ogni strato avrà larghezza 0.5
    print("\n--- Esempio 4: Ostacoli multipli distanti con 2 strati (larghezza singola 0.5) ---")
    obstacle_coords_4 = [
        [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)], # Ostacolo 1
        [(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)], # Ostacolo 2
        [(9, 2), (10, 2), (10, 3), (9, 3), (9, 2)] # Ostacolo 3
    ]
    ampiezza_capezzagna_4 = 0.5 # Larghezza di OGNI singolo strato
    num_strati_4 = 2
    all_capezzagne_layers_4, all_trajectories_layers_4 = calcola_capezzagne(
        obstacle_coords_4, ampiezza_capezzagna_4, num_concentric_layers=num_strati_4
    )

    enlarged_obstacles_4 = genera_ostacoli_ingranditi(obstacle_coords_4, all_capezzagne_layers_4)

    fig4, ax4 = plt.subplots(figsize=(8, 8))
    for i, obstacle_coords in enumerate(obstacle_coords_4):
        plot_polygon(ax4, Polygon(obstacle_coords), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale' if i == 0 else None)
    
    colors_4 = plt.cm.Greens(np.linspace(0.3, 0.9, num_strati_4))
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_4):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax4, capezzagna_poly, facecolor=colors_4[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 and i == 0 else None) # Solo la prima capezzagna del primo strato ha etichetta
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_4):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax4, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 and i == 0 else None) # Solo la prima traiettoria del primo strato ha etichetta

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_4):
        plot_polygon(ax4, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title(f"Capezzagne Concentriche, Traiettorie e Ostacoli Ingranditi (Esempio 4) - Larghezza Singolo Strato: {ampiezza_capezzagna_4}") # Corrected variable name
    ax4.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_4)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_4)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_4)}")


    # Esempio 5: Gruppi di ostacoli che si uniscono con 3 strati
    # Ogni strato avrà larghezza 0.4
    print("\n--- Esempio 5: Gruppi di ostacoli che si uniscono con 3 strati (larghezza singola 0.4) ---")
    obstacle_coords_5 = [
        # Gruppo 1
        [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
        [(2.2, 1.2), (3.2, 1.2), (3.2, 2.2), (2.2, 2.2), (2.2, 1.2)],
        # Gruppo 2
        [(6, 6), (7, 6), (7, 7), (6, 7), (6, 6)],
        [(7.2, 6.2), (8.2, 6.2), (8.2, 7.2), (7.2, 7.2), (7.2, 6.2)],
        # Ostacolo isolato
        [(10, 2), (11, 2), (11, 3), (10, 3), (10, 2)]
    ]
    ampiezza_capezzagna_5 = 0.4 # Larghezza di OGNI singolo strato
    num_strati_5 = 3
    all_capezzagne_layers_5, all_trajectories_layers_5 = calcola_capezzagne(
        obstacle_coords_5, ampiezza_capezzagna_5, num_concentric_layers=num_strati_5
    )

    enlarged_obstacles_5 = genera_ostacoli_ingranditi(obstacle_coords_5, all_capezzagne_layers_5)

    fig5, ax5 = plt.subplots(figsize=(8, 8))
    for i, obstacle_coords in enumerate(obstacle_coords_5):
        plot_polygon(ax5, Polygon(obstacle_coords), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale' if i == 0 else None)
    
    colors_5 = plt.cm.cividis(np.linspace(0.2, 0.8, num_strati_5))
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_5):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax5, capezzagna_poly, facecolor=colors_5[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_5):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax5, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_5):
        plot_polygon(ax5, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax5.set_aspect('equal', adjustable='box')
    ax5.set_title(f"Capezzagne Concentriche, Traiettorie e Ostacoli Ingranditi (Esempio 5) - Larghezza Singolo Strato: {ampiezza_capezzagna_5}")
    ax5.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_5)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_5)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_5)}")

    # --- Nuovo Esempio 6: Ostacolo a L con 3 strati di capezzagna ---
    # Ogni strato avrà larghezza 0.5
    print("\n--- Nuovo Esempio 6: Ostacolo a L con 3 strati di capezzagna (larghezza singola 0.5) ---")
    obstacle_coords_6 = [
        [(1,1), (5,1), (5,2), (2,2), (2,4), (1,4), (1,1)] # Ostacolo a L
    ]
    ampiezza_capezzagna_6 = 0.5 # Larghezza di OGNI singolo strato
    num_strati_6 = 3 # Numero di strati concentrici
    
    all_capezzagne_layers_6, all_trajectories_layers_6 = calcola_capezzagne(
        obstacle_coords_6, ampiezza_capezzagna_6, num_concentric_layers=num_strati_6
    )

    enlarged_obstacles_6 = genera_ostacoli_ingranditi(obstacle_coords_6, all_capezzagne_layers_6)

    fig6, ax6 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax6, Polygon(obstacle_coords_6[0]), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale')
    
    colors_6 = plt.cm.plasma(np.linspace(0.2, 0.8, num_strati_6))
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_6):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax6, capezzagna_poly, facecolor=colors_6[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_6):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax6, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_6):
        plot_polygon(ax6, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax6.set_aspect('equal', adjustable='box')
    ax6.set_title(f"Capezzagne Concentriche, Traiettorie e Ostacoli Ingranditi (Esempio 6) - Larghezza Singolo Strato: {ampiezza_capezzagna_6}")
    ax6.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_6)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_6)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_6)}")

    # --- Nuovo Esempio 7: Ostacolo irregolare complesso con 2 strati di capezzagna ---
    # Ogni strato avrà larghezza 0.6
    print("\n--- Nuovo Esempio 7: Ostacolo irregolare complesso con 2 strati di capezzagna (larghezza singola 0.6) ---")
    obstacle_coords_7 = [
        [(1,1), (3,0), (5,1), (4,3), (6,5), (3,6), (0,4), (1,1)] # Forma irregolare
    ]
    ampiezza_capezzagna_7 = 0.6 # Larghezza di OGNI singolo strato
    num_strati_7 = 2 # Numero di strati concentrici
    
    all_capezzagne_layers_7, all_trajectories_layers_7 = calcola_capezzagne(
        obstacle_coords_7, ampiezza_capezzagna_7, num_concentric_layers=num_strati_7
    )

    enlarged_obstacles_7 = genera_ostacoli_ingranditi(obstacle_coords_7, all_capezzagne_layers_7)

    fig7, ax7 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax7, Polygon(obstacle_coords_7[0]), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale')
    
    colors_7 = plt.cm.cool(np.linspace(0.2, 0.8, num_strati_7))
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_7):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax7, capezzagna_poly, facecolor=colors_7[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_7):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax7, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_7):
        plot_polygon(ax7, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax7.set_aspect('equal', adjustable='box')
    ax7.set_title(f"Capezzagne Concentriche, Traiettorie e Ostacoli Ingranditi (Esempio 7) - Larghezza Singolo Strato: {ampiezza_capezzagna_7}")
    ax7.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_7)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_7)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_7)}")

    # --- Nuovo Esempio 8: Test con num_concentric_layers = 0 (dovrebbe forzare a 2) ---
    print("\n--- Nuovo Esempio 8: Test con num_concentric_layers = 0 (dovrebbe forzare a 2) ---")
    obstacle_coords_8 = obstacle_coords_1 # Usa lo stesso ostacolo dell'esempio 1
    ampiezza_capezzagna_8 = 0.5 # Larghezza di OGNI singolo strato (se fossero 2)
    num_strati_8 = 0 # Test valore non valido
    
    all_capezzagne_layers_8, all_trajectories_layers_8 = calcola_capezzagne(
        obstacle_coords_8, ampiezza_capezzagna_8, num_concentric_layers=num_strati_8
    )

    enlarged_obstacles_8 = genera_ostacoli_ingranditi(obstacle_coords_8, all_capezzagne_layers_8)

    fig8, ax8 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax8, Polygon(obstacle_coords_8[0]), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale')
    
    # La funzione dovrebbe aver forzato num_strati_8 a 2, quindi adattiamo i colori
    colors_8 = plt.cm.viridis(np.linspace(0, 1, 2)) 
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_8):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax8, capezzagna_poly, facecolor=colors_8[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_8):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax8, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_8):
        plot_polygon(ax8, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax8.set_aspect('equal', adjustable='box')
    ax8.set_title(f"Capezzagne Forzate a 2 Strati e Ostacoli Ingranditi (Esempio 8) - Larghezza Singolo Strato: {ampiezza_capezzagna_8}")
    ax8.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_8)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_8)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_8)}")
    
    print("\n--- Nuovo Esempio 9: Test con num_concentric_layers = 1 (dovrebbe forzare a 2) ---")
    num_strati_9 = 1 # Test valore non valido
    
    all_capezzagne_layers_9, all_trajectories_layers_9 = calcola_capezzagne(
        obstacle_coords_8, ampiezza_capezzagna_8, num_concentric_layers=num_strati_9
    )

    enlarged_obstacles_9 = genera_ostacoli_ingranditi(obstacle_coords_8, all_capezzagne_layers_9)

    fig9, ax9 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax9, Polygon(obstacle_coords_8[0]), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale')
    
    # La funzione dovrebbe aver forzato num_strati_9 a 2, quindi adattiamo i colori
    colors_9 = plt.cm.viridis(np.linspace(0, 1, 2)) 
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_9):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax9, capezzagna_poly, facecolor=colors_9[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_9):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax9, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_9):
        plot_polygon(ax9, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax9.set_aspect('equal', adjustable='box')
    ax9.set_title(f"Capezzagne Forzate a 2 Strati e Ostacoli Ingranditi (Esempio 9) - Larghezza Singolo Strato: {ampiezza_capezzagna_8}")
    ax9.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_9)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_9)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_9)}")

    print("\n--- Nuovo Esempio 10: Test con num_concentric_layers = -5 (dovrebbe forzare a 2) ---")
    num_strati_10 = -5 # Test valore non valido
    
    all_capezzagne_layers_10, all_trajectories_layers_10 = calcola_capezzagne(
        obstacle_coords_8, ampiezza_capezzagna_8, num_concentric_layers=num_strati_10
    )

    enlarged_obstacles_10 = genera_ostacoli_ingranditi(obstacle_coords_8, all_capezzagne_layers_10)

    fig10, ax10 = plt.subplots(figsize=(8, 8))
    plot_polygon(ax10, Polygon(obstacle_coords_8[0]), facecolor='black', edgecolor='black', alpha=1.0, label='Ostacolo Originale')
    
    # La funzione dovrebbe aver forzato num_strati_10 a 2, quindi adattiamo i colori
    colors_10 = plt.cm.viridis(np.linspace(0, 1, 2)) 
    for i, layer_capezzagne in enumerate(all_capezzagne_layers_10):
        for j, capezzagna_poly in enumerate(layer_capezzagne):
            plot_polygon(ax10, capezzagna_poly, facecolor=colors_10[i], edgecolor='red', alpha=0.5, 
                         label=f'Capezzagna Strato {i+1}' if j == 0 else None)
    
    for i, layer_trajectories in enumerate(all_trajectories_layers_10):
        for j, trajectory_line in enumerate(layer_trajectories):
            plot_lines(ax10, [trajectory_line], color='green', linestyle='--', linewidth=2, 
                         label=f'Traiettoria Strato {i+1}' if j == 0 else None)

    for i, enlarged_obs_coords in enumerate(enlarged_obstacles_10):
        plot_polygon(ax10, Polygon(enlarged_obs_coords), facecolor='red', edgecolor='black', alpha=0.6, linewidth=3, linestyle='-',
                     label='Ostacolo Ingrandito' if i == 0 else None)

    ax10.set_aspect('equal', adjustable='box')
    ax10.set_title(f"Capezzagne Forzate a 2 Strati e Ostacoli Ingranditi (Esempio 10) - Larghezza Singolo Strato: {ampiezza_capezzagna_8}")
    ax10.legend()
    plt.grid(True)
    plt.show()
    print(f"Numero totale di strati di capezzagne: {len(all_capezzagne_layers_10)}")
    print(f"Numero totale di traiettorie: {len(all_trajectories_layers_10)}")
    print(f"Numero di ostacoli ingranditi generati: {len(enlarged_obstacles_10)}")