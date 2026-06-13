import tkinter as tk
import numpy as np
import math

def genera_traiettorie_con_ostacoli(confini_campo, larghezza_lavoro, angolo=0, ostacoli=None):
    """
    Genera traiettorie parallele all'interno di un campo rettangolare, evitando ostacoli.

    Args:
        confini_campo (tuple): (min_x, max_x, min_y, max_y) coordinate del campo.
        larghezza_lavoro (float): Larghezza dell'attrezzo di lavoro.
        angolo (float): Angolo in gradi delle traiettorie rispetto all'asse x.
        ostacoli (list of tuples): Lista di ostacoli, dove ogni ostacolo è definito
                                   come (min_x, max_x, min_y, max_y) del suo bounding box.

    Returns:
        list: Una lista di liste, dove ogni sottolista rappresenta una traiettoria
              come una sequenza di punti (x, y).
    """
    min_cx, max_cx, min_cy, max_cy = confini_campo
    traiettorie = []

    angolo_rad = np.radians(angolo)
    cos_theta = np.cos(angolo_rad)
    sin_theta = np.sin(angolo_rad)
    direzione = np.array([cos_theta, sin_theta])
    laterale = np.array([-sin_theta, cos_theta])

    punto_inizio = np.array([min_cx, min_cy])
    v1 = np.array([max_cx - min_cx, 0])
    v2 = np.array([0, max_cy - min_cy])
    lunghezza_v = max(np.dot(v1, laterale), np.dot(v2, laterale))
    num_passaggi = int(np.ceil(abs(lunghezza_v) / larghezza_lavoro))

    for i in range(num_passaggi + 1):
        punto_attuale = punto_inizio + laterale * i * larghezza_lavoro
        traiettoria = []
        t = 0
        while True:
            punto = punto_attuale + direzione * t
            x, y = punto

            # Verifica se il punto è all'interno dei confini del campo
            if not (min_cx <= x <= max_cx and min_cy <= y <= max_cy):
                break

            # Verifica se il punto interseca un ostacolo
            collisione = False
            if ostacoli:
                for ox_min, ox_max, oy_min, oy_max in ostacoli:
                    if ox_min <= x <= ox_max and oy_min <= y <= oy_max:
                        collisione = True
                        break

            if not collisione:
                traiettoria.append(punto.tolist())
                t += 0.1
            else:
                # Se colpisce un ostacolo, "salta" un po' per evitare di generare troppi punti vicini
                t += larghezza_lavoro / np.linalg.norm(direzione) # Salta di circa la larghezza

        if traiettoria:
            traiettorie.append(traiettoria)

    return traiettorie

def visualizza_campo_traiettorie_canvas(canvas, confini_campo, traiettorie, ostacoli, scale_x, scale_y, offset_x, offset_y):
    """Visualizza il campo, le traiettorie e gli ostacoli sul canvas."""
    min_x, max_x, min_y, max_y = confini_campo

    # Disegna i confini del campo
    points = [(min_x * scale_x + offset_x, min_y * scale_y + offset_y),
              (max_x * scale_x + offset_x, min_y * scale_y + offset_y),
              (max_x * scale_x + offset_x, max_y * scale_y + offset_y),
              (min_x * scale_x + offset_x, max_y * scale_y + offset_y)]
    canvas.create_polygon(points, outline='black', fill='', width=2, tags="campo")

    # Disegna gli ostacoli
    if ostacoli:
        for i, (ox_min, ox_max, oy_min, oy_max) in enumerate(ostacoli):
            obstacle_points = [(ox_min * scale_x + offset_x, oy_min * scale_y + offset_y),
                               (ox_max * scale_x + offset_x, oy_min * scale_y + offset_y),
                               (ox_max * scale_x + offset_x, oy_max * scale_y + offset_y),
                               (ox_min * scale_x + offset_x, oy_max * scale_y + offset_y)]
            canvas.create_polygon(obstacle_points, fill='brown', outline='black', tags=f"ostacolo_{i}")

    # Disegna le traiettorie
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, traiettoria in enumerate(traiettorie):
        if not traiettoria:
            continue
        points = [(p[0] * scale_x + offset_x, p[1] * scale_y + offset_y) for p in traiettoria]
        color = colors[i % len(colors)]
        canvas.create_line(points, fill=color, width=1, tags=f"traiettoria_{i}")

def main():
    window = tk.Tk()
    window.title("Simulazione Traiettorie con Ostacoli (Canvas)")

    canvas_width = 600
    canvas_height = 400
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack()

    # Variabili per memorizzare i parametri
    min_x_var = tk.StringVar(value="0")
    max_x_var = tk.StringVar(value="10")
    min_y_var = tk.StringVar(value="0")
    max_y_var = tk.StringVar(value="5")
    larghezza_var = tk.StringVar(value="1.5")
    angolo_var = tk.StringVar(value="0")
    ostacoli_str_var = tk.StringVar(value="[(2, 4, 1, 3), (6, 8, 0.5, 2.5)]") # Esempio di ostacoli

    def run_simulation():
        try:
            min_x = float(min_x_var.get())
            max_x = float(max_x_var.get())
            min_y = float(min_y_var.get())
            max_y = float(max_y_var.get())
            larghezza = float(larghezza_var.get())
            angolo = float(angolo_var.get())
            ostacoli_list_str = ostacoli_str_var.get()
            ostacoli = eval(ostacoli_list_str) # ATTENZIONE: eval può essere pericoloso con input non controllato

            confini = (min_x, max_x, min_y, max_y)
            traiettorie = genera_traiettorie_con_ostacoli(confini, larghezza, angolo, ostacoli)

            # Calcola scala e offset
            range_x = max_x - min_x
            range_y = max_y - min_y
            scale_x = canvas_width / range_x if range_x > 0 else 1
            scale_y = canvas_height / range_y if range_y > 0 else 1
            scale = min(scale_x, scale_y) * 0.8
            offset_x = (canvas_width - range_x * scale) / 2 - min_x * scale
            offset_y = (canvas_height - range_y * scale) / 2 - min_y * scale

            canvas.delete("all")
            visualizza_campo_traiettorie_canvas(canvas, confini, traiettorie, ostacoli, scale, scale, offset_x, offset_y)

        except ValueError:
            tk.messagebox.showerror("Errore", "Inserisci valori numerici validi per i confini e la larghezza.")
        except SyntaxError:
            tk.messagebox.showerror("Errore", "Formato degli ostacoli non valido. Usa [(min_x, max_x, min_y, max_y), ...]")

    # Interfaccia utente
    tk.Label(window, text="Confini Campo (min_x, max_x, min_y, max_y):").pack()
    frame_confini = tk.Frame(window)
    frame_confini.pack()
    tk.Entry(frame_confini, width=5, textvariable=min_x_var).pack(side=tk.LEFT)
    tk.Entry(frame_confini, width=5, textvariable=max_x_var).pack(side=tk.LEFT)
    tk.Entry(frame_confini, width=5, textvariable=min_y_var).pack(side=tk.LEFT)
    tk.Entry(frame_confini, width=5, textvariable=max_y_var).pack(side=tk.LEFT)

    tk.Label(window, text="Larghezza Lavoro:").pack()
    tk.Entry(window, textvariable=larghezza_var).pack()

    tk.Label(window, text="Angolo (gradi):").pack()
    tk.Entry(window, textvariable=angolo_var).pack()

    tk.Label(window, text="Ostacoli (lista di tuple (min_x, max_x, min_y, max_y)):").pack()
    entry_ostacoli = tk.Entry(window, textvariable=ostacoli_str_var, width=50)
    entry_ostacoli.pack()

    run_button = tk.Button(window, text="Simula", command=run_simulation)
    run_button.pack()

    window.mainloop()

if __name__ == "__main__":
    main()