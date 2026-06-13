import numpy as np
import random
import math
from shapely.geometry import Polygon, Point, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

# Assicurati che questo modulo esista e contenga la funzione `genera_ostacoli_poligonali_nell_area`
from genera_ostacoli_nell_area import genera_ostacoli_poligonali_nell_area

def genera_poligono_irregolare_generico(min_vertices=5, max_vertices=10, min_dim=20, max_dim=80):
    """
    Genera un singolo poligono irregolare con un numero casuale di vertici.
    Questo poligono sarà sempre un'area continua e unificata (senza buchi interni di sua natura).
    La concavità sarà data dalla disposizione dei vertici, non dalla sottrazione.
    """
    num_vertices = random.randint(min_vertices, max_vertices)
    
    # Reinizializzo la logica di generazione per garantire NO BUCHI
    final_polygon_points = []
    # Usiamo max_dim per determinare il range di generazione dei centri
    center_x_poly = random.uniform(max_dim / 2, 100 - max_dim / 2) # Esteso il range per centri
    center_y_poly = random.uniform(max_dim / 2, 100 - max_dim / 2)

    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
    
    # Raggi variabili per creare concavità/convessità
    base_radius = random.uniform(min_dim / 2, max_dim / 2)
    radius_variations = np.random.uniform(0.5, 1.5, num_vertices) # Variazione del raggio per irregolarità

    for i in range(num_vertices):
        r = base_radius * radius_variations[i]
        x = center_x_poly + r * np.cos(angles[i])
        y = center_y_poly + r * np.sin(angles[i])
        final_polygon_points.append((x, y))

    final_polygon = Polygon(final_polygon_points)

    # Sanifica il poligono finale se è invalido o vuoto
    if not final_polygon.is_valid:
        final_polygon = final_polygon.buffer(0) # Tenta di sanificare
    
    if isinstance(final_polygon, MultiPolygon):
        final_polygon = max(final_polygon.geoms, key=lambda p: p.area) # Prendi il più grande
        
    if not isinstance(final_polygon, Polygon) or final_polygon.is_empty or final_polygon.area < 10:
         # Riprova se il poligono è ancora invalido o troppo piccolo
         return genera_poligono_irregolare_generico(min_vertices, max_vertices, min_dim, max_dim)

    # Verifica se ci sono buchi interni non voluti generati dalla sanificazione
    if final_polygon.interiors:
        # Se ci sono buchi, riproviamo per garantire un'area unificata.
        return genera_poligono_irregolare_generico(min_vertices, max_vertices, min_dim, max_dim)
    
    return final_polygon


def genera_poligoni_irregolari_con_ostacoli(n_poligoni=300, max_ratio_ostacoli=0.3):
    dataset = []
    for _ in range(n_poligoni):
        field_polygon = genera_poligono_irregolare_generico()
        
        if not field_polygon or not field_polygon.is_valid or field_polygon.is_empty:
            continue

        area_coords = list(field_polygon.exterior.coords)
        
        # Num_obstacles_to_generate deve essere almeno 1, per garantire che ci siano ostacoli
        # altrimenti questo tipo di campo non rispetterebbe la sua caratteristica principale.
        # Ho modificato da random.randint(1, 3) a random.randint(1, 5) per una maggiore varietà.
        num_obstacles_to_generate = random.randint(1, 5) 
        
        ostacoli_generati_shapely = genera_ostacoli_poligonali_nell_area(
            area_coords=area_coords,
            n_ostacoli=num_obstacles_to_generate,
            max_attempts=1000,
            raggio_range=(2, 5),
            n_vertici_range=(5, 8),
            max_ratio_area=max_ratio_ostacoli
        )
        
        # Questa riga è quella che causava l'errore se ostacoli_generati_shapely fosse vuota
        # ma il check `if not o.is_empty` era corretto per gli oggetti Polygon.
        # Ora, `ostacoli_generati_shapely` è una lista di Polygon. Dobbiamo convertire
        # solo le coordinate.
        ostacoli = []
        for o in ostacoli_generati_shapely:
            if isinstance(o, Polygon) and not o.is_empty:
                ostacoli.append(list(o.exterior.coords))
            # Aggiungiamo un caso per gestire se per qualche motivo 'o' è già una lista di coordinate
            elif isinstance(o, list) and len(o) > 0 and isinstance(o[0], tuple): # Assumiamo sia una lista di tuple (coordinate)
                 ostacoli.append(o)
        

        dataset.append({
            "area_coords": area_coords,
            "obstacle_coords": ostacoli
        })

    return dataset

def salva_dataset(dataset, path="campi_con_ostacoli.npy"):
    np.save(path, dataset, allow_pickle=True)
    print(f"[✓] Dataset salvato: {path} ({len(dataset)} campi)")

if __name__ == "__main__":
    from shapely.geometry import Point # Per test della funzione irregolare
    
    print("Test generazione di campi irregolari con ostacoli (solo forma, no buchi):")
    test_dataset = genera_poligoni_irregolari_con_ostacoli(n_poligoni=5, max_ratio_ostacoli=0.2)
    for i, campo in enumerate(test_dataset):
        print(f"Campo {i+1}:")
        print(f"  Area Vertici: {len(campo['area_coords'])}")
        print(f"  Ostacoli: {len(campo['obstacle_coords'])}")
        field_poly = Polygon(campo['area_coords'])
        print(f"  Area del poligono: {field_poly.area:.2f}")
        print(f"  Ha buchi interni? {bool(field_poly.interiors)}") # Deve essere False
        is_trapezoidal_check = len(field_poly.exterior.coords) == 5 and field_poly.is_convex
        print(f"  È un trapezio (check euristico)? {is_trapezoidal_check}") # Dovrebbe essere False