import numpy as np
import random
import math
from shapely.geometry import Polygon, MultiPolygon # Aggiunto MultiPolygon

def genera_poligono_irregolare_generico(min_vertices=5, max_vertices=10, min_dim=20, max_dim=80):
    """
    Genera un singolo poligono irregolare con un numero casuale di vertici.
    Questo poligono sarà sempre un'area continua e unificata (senza buchi interni di sua natura).
    La concavità sarà data dalla disposizione dei vertici, non dalla sottrazione.
    """
    num_vertices = random.randint(min_vertices, max_vertices)
    
    final_polygon_points = []
    # Usiamo max_dim per determinare il range di generazione dei centri
    center_x_poly = random.uniform(max_dim / 2, 100 - max_dim / 2) 
    center_y_poly = random.uniform(max_dim / 2, 100 - max_dim / 2) 

    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
    
    base_radius = random.uniform(min_dim / 2, max_dim / 2)
    radius_variations = np.random.uniform(0.5, 1.5, num_vertices)

    for i in range(num_vertices):
        r = base_radius * radius_variations[i]
        x = center_x_poly + r * np.cos(angles[i])
        y = center_y_poly + r * np.sin(angles[i])
        final_polygon_points.append((x, y))

    final_polygon = Polygon(final_polygon_points)

    if not final_polygon.is_valid:
        final_polygon = final_polygon.buffer(0) # Tenta di sanificare
    
    if isinstance(final_polygon, MultiPolygon):
        final_polygon = max(final_polygon.geoms, key=lambda p: p.area) # Prendi il più grande
        
    if not isinstance(final_polygon, Polygon) or final_polygon.is_empty or final_polygon.area < 10:
         # Riprova se il poligono è ancora invalido o troppo piccolo
         return genera_poligono_irregolare_generico(min_vertices, max_vertices, min_dim, max_dim)
    
    if final_polygon.interiors: # Se ci sono buchi interni non voluti generati dalla sanificazione
        return genera_poligono_irregolare_generico(min_vertices, max_vertices, min_dim, max_dim)

    return final_polygon


def genera_poligoni_irregolari_senza_ostacoli(n_poligoni=300):
    dataset = []
    attempts = 0
    generated_count = 0
    max_attempts_per_field = 100 # Limite per evitare loop infiniti nella generazione di un singolo campo

    while generated_count < n_poligoni and attempts < n_poligoni * max_attempts_per_field:
        field_polygon = genera_poligono_irregolare_generico()

        if not field_polygon or not field_polygon.is_valid or field_polygon.is_empty:
            attempts += 1
            continue

        area_coords = list(field_polygon.exterior.coords)
        
        dataset.append({
            "area": np.array(area_coords),
            "ostacoli": []  # nessun ostacolo
        })
        generated_count += 1
        attempts = 0 # Reset attempts for next field if successful

    if generated_count < n_poligoni:
        print(f"[!] Warning: Generati solo {generated_count} campi irregolari senza ostacoli su {n_poligoni} richiesti.")

    return dataset


def salva_dataset(dataset, path="campi_senza_ostacoli.npy"):
    np.save(path, dataset, allow_pickle=True)
    print(f"[✓] Dataset salvato: {path} ({len(dataset)} campi)")


if __name__ == "__main__":
    print("Test generazione di campi irregolari senza ostacoli (solo forma, no buchi):")
    data = genera_poligoni_irregolari_senza_ostacoli(n_poligoni=5)
    salva_dataset(data)

    for i, campo in enumerate(data):
        print(f"Campo {i+1}:")
        print(f"  Area Vertici: {len(campo['area'])}")
        print(f"  Ostacoli: {len(campo['ostacoli'])}")
        field_poly = Polygon(campo['area'])
        print(f"  Area del poligono: {field_poly.area:.2f}")
        print(f"  Ha buchi interni? {bool(field_poly.interiors)}") # Deve essere False
        # Nota: un campo irregolare per definizione non dovrebbe essere un trapezio
        is_trapezoidal_check = len(field_poly.exterior.coords) == 5 and field_poly.is_valid # is_convex removed
        print(f"  È un trapezio (check euristico)? {is_trapezoidal_check}") # Dovrebbe essere False