import math
import random
from shapely.geometry import Polygon, box

def genera_ostacoli_poligonali_nell_area(area_coords, n_ostacoli=30, max_attempts=1000,
                                          raggio_range=(3, 8), n_vertici_range=(5, 8),
                                          max_ratio_area=0.3):
    """
    Genera ostacoli poligonali chiusi contenuti nell'area specificata,
    con area totale massima pari a max_ratio_area * area_cella.
    """
    area_poly = Polygon(area_coords)
    minx, miny, maxx, maxy = area_poly.bounds
    area_totale = area_poly.area

    ostacoli_shapely = [] # Lista di oggetti Polygon
    area_ostacoli_attuali = 0.0
    tentativi = 0

    while len(ostacoli_shapely) < n_ostacoli and tentativi < max_attempts:
        # Genera un centro casuale all'interno del bounding box dell'area, con un margine
        # per evitare che gli ostacoli siano troppo vicini al bordo e causino problemi
        # o non siano completamente contenuti.
        margin = max(raggio_range) # Margine basato sul raggio massimo dell'ostacolo
        cx = random.uniform(minx + margin, maxx - margin)
        cy = random.uniform(miny + margin, maxy - margin)
        
        n_vert = random.randint(*n_vertici_range)
        r_base = random.uniform(*raggio_range)

        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(n_vert)])
        punti = [
            (
                cx + r_base * random.uniform(0.7, 1.3) * math.cos(a),
                cy + r_base * random.uniform(0.7, 1.3) * math.sin(a)
            )
            for a in angles
        ]
        
        poligono = Polygon(punti) # Crea il poligono senza ripetere il primo punto per la chiusura

        # Controlla validità e contenimento
        if poligono.is_valid and not poligono.is_empty and area_poly.contains(poligono):
            # Controlla che non si sovrapponga a ostacoli già aggiunti
            overlaps_existing = False
            for existing_obs in ostacoli_shapely:
                if poligono.intersects(existing_obs.buffer(0.01)): # Buffer per evitare tangenze problematiche
                    overlaps_existing = True
                    break
            
            if not overlaps_existing:
                area_attuale = poligono.area
                if area_ostacoli_attuali + area_attuale <= area_totale * max_ratio_area:
                    ostacoli_shapely.append(poligono) # Append the shapely Polygon object directly
                    area_ostacoli_attuali += area_attuale
        
        tentativi += 1

    if len(ostacoli_shapely) < n_ostacoli and n_ostacoli > 0: # Solo se si volevano ostacoli e non si sono raggiunti
        print(f"[!] Generati solo {len(ostacoli_shapely)} ostacoli su {n_ostacoli} (occupazione: {area_ostacoli_attuali:.2f} / {area_totale * max_ratio_area:.2f})")

    return ostacoli_shapely # Ritorna la lista di oggetti Polygon