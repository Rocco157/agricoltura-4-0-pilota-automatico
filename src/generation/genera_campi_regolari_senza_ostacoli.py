import numpy as np
import random
from shapely.geometry import Polygon

def genera_trapezi_irregolari(n_campi=300,
                              base1_range=(20, 50),
                              base2_ratio_range=(0.6, 1.4),
                              altezza_range=(20, 50),
                              shift_range=(-10, 10)) -> list:
    """
    Genera campi trapezoidali irregolari.
    Ogni campo è definito da 4 vertici + chiusura come numpy array.

    Params:
        n_campi (int): numero di campi da generare
        base1_range (tuple): lunghezza minima e massima della base inferiore
        base2_ratio_range (tuple): ratio tra base superiore e base1
        altezza_range (tuple): altezza del trapezio
        shift_range (tuple): spostamento della base superiore in x

    Returns:
        list di np.ndarray: ciascuna array di shape (N,2) con i vertici del poligono
    """
    dataset = []
    attempts = 0
    generated_count = 0
    max_attempts_per_field = 100 # Limite per evitare loop infiniti nella generazione di un singolo campo

    while generated_count < n_campi and attempts < n_campi * max_attempts_per_field:
        x0 = random.uniform(0, 100)
        y0 = random.uniform(0, 100)
        base1 = random.uniform(*base1_range)
        ratio = random.uniform(*base2_ratio_range)
        base2 = base1 * ratio
        h = random.uniform(*altezza_range)
        dx = random.uniform(*shift_range)

        p1 = (x0, y0)
        p2 = (x0 + base1, y0)
        p3 = (x0 + dx + base2, y0 + h)
        p4 = (x0 + dx, y0 + h)

        poly_coords = [p1, p2, p3, p4, p1] # Chiude il poligono
        
        # Validazione base: assicurati che il poligono sia valido e non degenere
        temp_polygon = Polygon(poly_coords)
        if not temp_polygon.is_valid or temp_polygon.is_empty or temp_polygon.area < 10: # Area minima
            attempts += 1
            continue

        dataset.append({
            "area": np.array(poly_coords),
            "ostacoli": [] # nessun ostacolo
        })
        generated_count += 1
        attempts = 0 # Reset attempts for next field if successful
    
    if generated_count < n_campi:
        print(f"[!] Warning: Generati solo {generated_count} campi regolari su {n_campi} richiesti.")

    return dataset


def salva_dataset(dataset, path="campi_regolari_senza_ostacoli.npy"):
    np.save(path, dataset, allow_pickle=True)
    print(f"[✓] Dataset salvato: {path} ({len(dataset)} campi)")


if __name__ == "__main__":
    print("Test generazione di campi regolari senza ostacoli:")
    data = genera_trapezi_irregolari(n_campi=5)
    salva_dataset(data)

    for i, campo in enumerate(data):
        print(f"Campo {i+1}:")
        print(f"  Area Vertici: {len(campo['area'])}")
        print(f"  Ostacoli: {len(campo['ostacoli'])}")
        field_poly = Polygon(campo['area'])
        print(f"  Area del poligono: {field_poly.area:.2f}")
        print(f"  È un trapezio (check euristico)? {len(field_poly.exterior.coords) == 5 and field_poly.is_valid}") # is_convex removed