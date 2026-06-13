import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
import random
import warnings

# Importa i tuoi moduli di generazione campi, ora aggiornati
from genera_campi_con_ostacoli import genera_poligoni_irregolari_con_ostacoli # Genera campi irregolari (non trapezi) con ostacoli
from genera_campi_irregolari_senza_ostacoli import genera_poligoni_irregolari_senza_ostacoli # Genera campi irregolari (non trapezi) senza ostacoli
from genera_campi_regolari_senza_ostacoli import genera_trapezi_irregolari # Genera campi regolari (trapezi) senza ostacoli

# Importa i moduli di utility
from genera_capezzagne_ostacoli import calcola_capezzagne
from genera_capezzagne_bordocampo import HeadlandCalculator
from decomposizione_area import AreaCoverageProblem
from genera_traiettoria_sottocella import genera_traiettorie_parallele, genera_traiettorie_svolta

# Parametri di generazione globali
NUM_CAMPI_PER_TIPOLOGIA = 100 # Nuovo requisito: 100 di ogni tipo

NUM_CAPEZZAGNE_OSTACOLI = 2 # Strati di capezzagne attorno agli ostacoli
NUM_CAPEZZAGNE_BORDOCAMPO = 2 # Strati di capezzagne a bordo campo
AMPIEZZA_LAVORO = 2.0 # Larghezza di lavoro del mezzo (es. 2 metri)
LUNGHEZZA_MEZZO = 5.0 # Lunghezza del mezzo per calcolare la svolta (es. 5 metri)
RAGGIO_SVOLTA_MINIMO = 3.0 # Raggio di svolta minimo del mezzo (es. 3 metri)

MAX_ATTEMPTS_PER_FIELD = 100 # Numero massimo di tentativi per generare un singolo campo valido
MAX_ATTEMPTS_OVERALL = NUM_CAMPI_PER_TIPOLOGIA * MAX_ATTEMPTS_PER_FIELD # Limite totale per la generazione del dataset
# Per gestire avvertimenti da Shapely o altri moduli
warnings.filterwarnings("ignore", category=UserWarning, module="shapely")
warnings.filterwarnings("ignore", category=FutureWarning)

def run_workflow_for_field(area_coords, obstacle_coords, ampi_lavoro, field_type):
    """
    Esegue l'intero workflow di generazione del percorso per un singolo campo
    in base alla sua tipologia.
    Restituisce le features del campo e l'orientamento della traiettoria come label.
    """
    original_area_polygon = Polygon(area_coords)
    
    # Validazione iniziale del poligono dell'area
    if not original_area_polygon.is_valid or original_area_polygon.is_empty or original_area_polygon.area == 0:
        # print(f"Skipping invalid/empty field polygon with area {original_area_polygon.area}.")
        return None, None 

    # Determina le caratteristiche del campo per le features
    is_irregular_shape = not (len(area_coords) == 5) # Euristica per trapezio (4 vertici unici + chiusura)
    is_irregular_obstacles = len(obstacle_coords) > 0
    is_irregular_field_total = is_irregular_shape or is_irregular_obstacles
    
    field_features = {}
    field_features['original_area_area'] = original_area_polygon.area
    field_features['original_area_perimeter'] = original_area_polygon.length
    minx, miny, maxx, maxy = original_area_polygon.bounds
    field_features['original_area_bbox_minx'] = minx
    field_features['original_area_bbox_miny'] = miny
    field_features['original_area_bbox_maxx'] = maxx
    field_features['original_area_bbox_maxy'] = maxy
    field_features['original_area_width'] = maxx - minx
    field_features['original_area_height'] = maxy - miny
    field_features['original_area_aspect_ratio'] = (maxx - minx) / (maxy - miny) if (maxy - miny) != 0 else 1.0
    field_features['num_vertices_area'] = len(area_coords)
    field_features['num_obstacles'] = len(obstacle_coords)
    
    total_obstacle_area = sum(Polygon(obs).area for obs in obstacle_coords) if obstacle_coords else 0.0
    field_features['total_obstacle_area'] = total_obstacle_area
    field_features['obstacle_area_ratio'] = total_obstacle_area / original_area_polygon.area if original_area_polygon.area > 0 else 0.0

    field_features['is_irregular_field'] = 1 if is_irregular_field_total else 0
    field_features['is_irregular_shape'] = 1 if is_irregular_shape else 0
    field_features['has_obstacles'] = 1 if is_irregular_obstacles else 0
    
    # Aggiungi una feature per il tipo di campo esplicito
    field_features['field_type'] = field_type 

    current_working_area = original_area_polygon

    # Gestione delle capezzagne e dell'area di lavoro in base al tipo di campo
    field_features['area_capezzagne_ostacoli'] = 0.0
    field_features['area_capezzagne_bordocampo'] = 0.0

    if field_type == 'irregolare_con_ostacoli':
        # 1. Calcolo capezzagne attorno agli ostacoli
        if is_irregular_obstacles:
            try:
                all_capezzagne_layers, _ = calcola_capezzagne(obstacle_coords, ampi_lavoro, NUM_CAPEZZAGNE_OSTACOLI)
                for layer_polygons in all_capezzagne_layers:
                    for poly in layer_polygons:
                        if poly.is_valid and not poly.is_empty:
                            current_working_area = current_working_area.difference(poly)
                field_features['area_capezzagne_ostacoli'] = sum(p.area for layer in all_capezzagne_layers for p in layer if p.is_valid and not p.is_empty)
            except Exception as e:
                # print(f"Warning (irregolare_con_ostacoli): Failed to calculate obstacle headlands: {e}")
                return None, None # Se fallisce, il campo non è utilizzabile

        # 2. Calcolo capezzagne a bordo campo
        try:
            hc_initial = HeadlandCalculator(area_coords, obstacle_coords) # Passiamo gli ostacoli originali per il calcolo del bordo
            headland_trajectories_bordocampo_initial, _ = hc_initial.calculate_headlands(ampi_lavoro, NUM_CAPEZZAGNE_BORDOCAMPO)

            capezzagne_bordocampo_polygons = []
            for traj_line in headland_trajectories_bordocampo_initial:
                if not traj_line.is_empty:
                    buffer_poly = traj_line.buffer(ampi_lavoro / 2.0, cap_style=3, join_style=2)
                    if buffer_poly.is_valid and not buffer_poly.is_empty:
                        capezzagne_bordocampo_polygons.append(buffer_poly)

            if capezzagne_bordocampo_polygons:
                unified_bordocampo = unary_union(capezzagne_bordocampo_polygons)
                if unified_bordocampo.is_valid and not unified_bordocampo.is_empty:
                    current_working_area = current_working_area.difference(unified_bordocampo)
            
            field_features['area_capezzagne_bordocampo'] = sum(p.area for p in capezzagne_bordocampo_polygons if p.is_valid and not p.is_empty)
        except Exception as e:
            # print(f"Warning (irregolare_con_ostacoli): Failed to calculate border headlands: {e}")
            return None, None # Se fallisce, il campo non è utilizzabile

    elif field_type == 'irregolare_senza_ostacoli':
        # Solo capezzagne a bordo campo
        try:
            hc_initial = HeadlandCalculator(area_coords, obstacle_coords) # obstacle_coords sarà vuoto
            headland_trajectories_bordocampo_initial, _ = hc_initial.calculate_headlands(ampi_lavoro, NUM_CAPEZZAGNE_BORDOCAMPO)

            capezzagne_bordocampo_polygons = []
            for traj_line in headland_trajectories_bordocampo_initial:
                if not traj_line.is_empty:
                    buffer_poly = traj_line.buffer(ampi_lavoro / 2.0, cap_style=3, join_style=2)
                    if buffer_poly.is_valid and not buffer_poly.is_empty:
                        capezzagne_bordocampo_polygons.append(buffer_poly)

            if capezzagne_bordocampo_polygons:
                unified_bordocampo = unary_union(capezzagne_bordocampo_polygons)
                if unified_bordocampo.is_valid and not unified_bordocampo.is_empty:
                    current_working_area = current_working_area.difference(unified_bordocampo)
            
            field_features['area_capezzagne_bordocampo'] = sum(p.area for p in capezzagne_bordocampo_polygons if p.is_valid and not p.is_empty)
        except Exception as e:
            # print(f"Warning (irregolare_senza_ostacoli): Failed to calculate border headlands: {e}")
            return None, None

    elif field_type == 'regolare_senza_ostacoli':
        # Solo capezzagne a bordo campo (come irregolare_senza_ostacoli)
        try:
            hc_initial = HeadlandCalculator(area_coords, obstacle_coords) # obstacle_coords sarà vuoto
            headland_trajectories_bordocampo_initial, _ = hc_initial.calculate_headlands(ampi_lavoro, NUM_CAPEZZAGNE_BORDOCAMPO)

            capezzagne_bordocampo_polygons = []
            for traj_line in headland_trajectories_bordocampo_initial:
                if not traj_line.is_empty:
                    buffer_poly = traj_line.buffer(ampi_lavoro / 2.0, cap_style=3, join_style=2)
                    if buffer_poly.is_valid and not buffer_poly.is_empty:
                        capezzagne_bordocampo_polygons.append(buffer_poly)

            if capezzagne_bordocampo_polygons:
                unified_bordocampo = unary_union(capezzagne_bordocampo_polygons)
                if unified_bordocampo.is_valid and not unified_bordocampo.is_empty:
                    current_working_area = current_working_area.difference(unified_bordocampo)
            
            field_features['area_capezzagne_bordocampo'] = sum(p.area for p in capezzagne_bordocampo_polygons if p.is_valid and not p.is_empty)
        except Exception as e:
            # print(f"Warning (regolare_senza_ostacoli): Failed to calculate border headlands: {e}")
            return None, None
    else:
        # Questo caso non dovrebbe verificarsi con le tipologie definite
        return None, None


    # 3. Decomposizione trapezoidale
    cells = []
    # Assicurati che l'area sia valida e significativa per la decomposizione
    if not current_working_area.is_empty and current_working_area.is_valid and current_working_area.area > 0.01:
        try:
            if isinstance(current_working_area, (MultiPolygon, GeometryCollection)):
                # Itera sui singoli poligoni se è un MultiPolygon o GeometryCollection
                for geom in current_working_area.geoms:
                    if isinstance(geom, Polygon) and not geom.is_empty and geom.is_valid and geom.area > 0.01:
                        problem = AreaCoverageProblem(list(geom.exterior.coords), [list(interior.coords) for interior in geom.interiors])
                        problem.decomponi_area()
                        cells.extend(problem.cells)
            elif isinstance(current_working_area, Polygon):
                # Se è un singolo Polygon
                problem_area_coords = list(current_working_area.exterior.coords)
                problem_obstacle_coords = [list(interior.coords) for interior in current_working_area.interiors]
                problem = AreaCoverageProblem(problem_area_coords, problem_obstacle_coords)
                problem.decomponi_area()
                cells = problem.cells
        except Exception as e:
            # print(f"Warning: Failed decomposition for area: {e}")
            pass # cells will remain empty

    field_features['num_decomposed_cells'] = len(cells)
    
    total_trajectory_length = 0.0
    first_cell_row_dir = np.array([0.0, 0.0]) # Default a zero
    has_valid_trajectory = False

    for i, cell in enumerate(cells):
        if isinstance(cell, Polygon) and not cell.is_empty and cell.is_valid and cell.area > 0.01:
            try:
                # Per i campi regolari (trapezoidali senza ostacoli), genera traiettorie sull'unica sottocella
                # Per gli altri, genera su tutte le sottocelle risultanti dalla decomposizione
                trajectories_file, row_dir_current_cell, _ = genera_traiettorie_parallele(cell, ampi_lavoro, orientamento_xy=None)
                
                if not has_valid_trajectory and np.linalg.norm(row_dir_current_cell) > 0:
                    first_cell_row_dir = row_dir_current_cell
                    has_valid_trajectory = True
                
                for traj in trajectories_file:
                    total_trajectory_length += traj.length
            except Exception as e:
                # print(f"Warning: Failed to generate trajectory for cell {i} with error: {e}")
                pass
        
    field_features['total_trajectory_length'] = total_trajectory_length
    
    # Se non è stata trovata alcuna traiettoria valida, l'orientamento rimane [0,0]
    trajectory_orientation = tuple(first_cell_row_dir)

    field_features['area_coords'] = str(area_coords)
    field_features['obstacle_coords'] = str(obstacle_coords)

    return field_features, trajectory_orientation

def generate_dataset(num_samples_per_type=NUM_CAMPI_PER_TIPOLOGIA):
    """
    Genera un dataset di campi agricoli con le features e le labels, producendo
    un numero fisso di campioni per ogni tipologia.
    """
    dataset_records = []
    
    print(f"Inizio generazione del dataset (total {num_samples_per_type * 3} campi stimati)...")

    # Tipo 1: Campi Regolari (Trapezoidali, Senza Ostacoli)
    print("Generazione di 100 campi regolari (trapezoidali senza ostacoli)...")
    generated_count = 0
    attempts = 0
    while generated_count < num_samples_per_type and attempts < MAX_ATTEMPTS_OVERALL:
        field_data_list = genera_trapezi_irregolari(n_campi=1)
        if not field_data_list: # Assicurati che non sia vuota
            attempts += 1
            continue

        # ACCESSO CORRETTO: Recupera le coordinate dal dizionario e poi converti in lista
        area_coords_raw = field_data_list[0]['area'].tolist()
        obstacle_coords = [] # Nessun ostacolo per questo tipo di campo

        features, label_orientation = run_workflow_for_field(area_coords_raw, obstacle_coords, AMPIEZZA_LAVORO, 'regolare_senza_ostacoli')
        if features is not None and label_orientation is not None:
            features['label_orientation_x'] = label_orientation[0]
            features['label_orientation_y'] = label_orientation[1]
            dataset_records.append(features)
            generated_count += 1
            if generated_count % 10 == 0:
                print(f"  Generati {generated_count}/{num_samples_per_type} campi regolari (senza ostacoli).")
        attempts += 1
    print(f"Completato generazione campi regolari (senza ostacoli). Campi validi generati: {generated_count}")


    # Tipo 2: Campi Irregolari per Forma (Non Trapezoidali, Senza Ostacoli)
    print(f"\nGenerazione di {num_samples_per_type} campi irregolari per la forma (non trapezoidali, senza ostacoli)...")
    generated_count = 0
    attempts = 0
    while generated_count < num_samples_per_type and attempts < num_samples_per_type * 5:
        field_data_list = genera_poligoni_irregolari_senza_ostacoli(n_poligoni=1)
        if not field_data_list:
            attempts += 1
            continue
        area_coords_raw = field_data_list[0]['area'].tolist()
        obstacle_coords = [] # Nessun ostacolo

        features, label_orientation = run_workflow_for_field(area_coords_raw, obstacle_coords, AMPIEZZA_LAVORO, 'irregolare_senza_ostacoli')
        if features is not None and label_orientation is not None:
            features['label_orientation_x'] = label_orientation[0]
            features['label_orientation_y'] = label_orientation[1]
            dataset_records.append(features)
            generated_count += 1
            if generated_count % 10 == 0:
                print(f"  Generati {generated_count}/{num_samples_per_type} campi irregolari (solo forma).")
        attempts += 1
    print(f"Completato generazione campi irregolari (solo forma). Campi validi generati: {generated_count}")

    # Tipo 3: Campi Irregolari per Forma (Non Trapezoidali, Con Ostacoli)
    print("\nGenerazione di 100 campi irregolari (con ostacoli)...")
    generated_count = 0
    total_attempts_for_type = 0 # Contatore corretto per questa tipologia
    while generated_count < num_samples_per_type and total_attempts_for_type < MAX_ATTEMPTS_OVERALL:
        field_data_list = genera_poligoni_irregolari_con_ostacoli(n_poligoni=1)

        if not field_data_list: # Se la generazione del campo fallisce all'origine
            total_attempts_for_type += 1
            continue

        field_data = field_data_list[0]
        # MODIFICA ESSENZIALE QUI: NON USARE .tolist()
        area_coords_raw = field_data['area_coords'] # <--- CORREZIONE DEFINITIVA
        obstacle_coords = field_data['obstacle_coords'] # Questa è già corretta

        # Verifica che ci siano ostacoli e che l'area rimanente sia significativa
        # Assicurati che Polygon venga importato da shapely.geometry
        try:
            original_area_poly = Polygon(area_coords_raw)
            original_area = original_area_poly.area
        except Exception as e:
            # Gestisci il caso in cui area_coords_raw sia invalido per Polygon
            print(f"Warning: Impossibile creare poligono dall'area: {e}")
            total_attempts_for_type += 1
            continue


        if not obstacle_coords:
            total_attempts_for_type += 1
            continue

        # Calcola l'area rimanente sottraendo gli ostacoli per validazione
        temp_area_poly = original_area_poly # Usa il poligono già creato
        is_valid_field = True
        for obs_coords in obstacle_coords:
            try:
                obs_poly = Polygon(obs_coords)
                if not obs_poly.is_valid:
                    is_valid_field = False
                    break
                if temp_area_poly.intersects(obs_poly):
                    temp_area_poly = temp_area_poly.difference(obs_poly)
            except Exception as e:
                print(f"Warning: Errore nella gestione dell'ostacolo: {e}")
                is_valid_field = False
                break
        
        if not is_valid_field:
            total_attempts_for_type += 1
            continue

        remaining_area = temp_area_poly.area if temp_area_poly.is_valid else 0

        # Se l'area rimanente è troppo piccola (gli ostacoli "coprono" troppo)
        if remaining_area < 0.1 * original_area: # Gli ostacoli devono lasciare un'area lavorabile
            total_attempts_for_type += 1
            continue
        
        # Esegui il workflow solo se il campo e gli ostacoli sono validi
        features, label_orientation = run_workflow_for_field(area_coords_raw, obstacle_coords, AMPIEZZA_LAVORO, 'irregolare_con_ostacoli')
        if features is not None and label_orientation is not None:
            features['label_orientation_x'] = label_orientation[0]
            features['label_orientation_y'] = label_orientation[1]
            dataset_records.append(features)
            generated_count += 1
            if generated_count % 10 == 0:
                print(f"  Generati {generated_count}/{num_samples_per_type} campi irregolari (con ostacoli).")
        total_attempts_for_type += 1 # Incrementa i tentativi complessivi per questa tipologia, sia che il campo sia valido o meno.
    print(f"Completato generazione campi irregolari (con ostacoli). Campi validi generati: {generated_count}")

  
    
    df = pd.DataFrame(dataset_records)
    return df

if __name__ == "__main__":
    df_dataset = generate_dataset(NUM_CAMPI_PER_TIPOLOGIA)
    output_filename = "agricultural_field_path_planning_dataset.csv"
    df_dataset.to_csv(output_filename, index=False)
    print(f"\nDataset generato con successo e salvato in '{output_filename}'")
    print(df_dataset.head())
    print(f"Dimensioni del dataset: {df_dataset.shape}")
    
    # Conteggio per tipo per verificare il bilanciamento
    print("\nConteggio dei campi generati per tipologia:")
    print(df_dataset['field_type'].value_counts())