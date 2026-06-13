# 🌾 Agricoltura 4.0: Sistemi di pilota automatico per lavorazioni in campo aperto

> Tesi di Laurea Triennale in Ingegneria Informatica  
> Università della Calabria — A.A. 2024/2025  
> Candidato: Rocco Martino | Relatore: Prof. Sergio Flesca | Correlatore: Ing. Francesco Scala

---

## 📌 Panoramica

Questo progetto implementa un **framework completo per la pianificazione intelligente delle traiettorie in campo agricolo**, combinando algoritmi geometrici ed euristici con una componente di rete neurale (**FieldNet**) a supporto delle decisioni.

Il sistema è in grado di:
- generare automaticamente scenari agricoli sintetici (regolari, irregolari, con ostacoli);
- calcolare le **capezzagne** attorno ai bordi del campo e agli ostacoli;
- decomporre l’area di lavoro in sotto-celle tramite **decomposizione trapezoidale**;
- generare traiettorie **boustrofediche** (andata e ritorno) per ciascuna sotto-cella, con svolte smussate tramite curve di Bézier;
- addestrare una **rete neurale multi-task** (FieldNet) per classificare la tipologia del campo e prevedere l’orientamento ottimale di lavoro.

---

## 🗂️ Struttura del repository

```text
Sviluppi_Tesi/
├── data/
│   └── agricultural_field_path_planning_dataset.csv
│
├── src/
│   ├── geometry/
│   │   ├── decomposizione_area.py
│   │   └── decomp_area_ampiezza.py
│   ├── generation/
│   │   ├── dataset_generator.py
│   │   ├── genera_campi_con_ostacoli.py
│   │   ├── genera_campi_irregolari_senza_ostacoli.py
│   │   ├── genera_campi_regolari_senza_ostacoli.py
│   │   ├── genera_capezzagne.py
│   │   ├── genera_capezzagne_bordocampo.py
│   │   ├── genera_capezzagne_ostacoli.py
│   │   ├── genera_ostacoli_nell_area.py
│   │   └── genera_filari.py
│   ├── planning/
│   │   ├── traiettoria_ideale.py
│   │   └── genera_traiettoria_sottocella.py
│   ├── ml/
│   │   └── rete_neurale_predittiva.py
│   └── viz/
│       ├── visualizza_csv.py
│       └── genera_grafico.py
│
└── requirements.txt
```

---

## ⚙️ Pipeline

```text
genera_campi_*.py
       │
       ▼
dataset_generator.py  ──►  agricultural_field_path_planning_dataset.csv
       │
       ├── genera_capezzagne_*.py   (calcolo delle capezzagne)
       ├── decomposizione_area.py   (decomposizione trapezoidale)
       └── genera_traiettoria_sottocella.py  (boustrophedon + Bézier)
                                    │
                                    ▼
                        rete_neurale_predittiva.py
                          (addestramento e valutazione di FieldNet)
```

---

## 🧠 FieldNet — Architettura della rete neurale

FieldNet è una **Deep Neural Network multi-task** realizzata con PyTorch e addestrata sul dataset generato automaticamente.

| Componente | Dettagli |
|------------|----------|
| Input | Area, perimetro, larghezza, altezza, rapporto d’aspetto, numero di vertici, metriche sugli ostacoli, aree di capezzagna, numero di sotto-celle, lunghezza totale della traiettoria |
| Strato condiviso | 2 layer Linear + ReLU |
| Testa di regressione | Predizione dell’orientamento ottimale di lavoro (coordinate X, Y) |
| Testa di classificazione | Classificazione della tipologia del campo (regolare / irregolare / con ostacoli) |
| Loss | MSELoss per la regressione + CrossEntropyLoss per la classificazione |
| Ottimizzatore | Adam, learning rate 0.001 |
| Addestramento | 100 epoche, batch size 32, split training/validation 80/20 |
| File di output | `fieldnet_multitask.pt` |

---

## 📊 Dataset

Il dataset è **sintetico e generato automaticamente** tramite `dataset_generator.py`.

| Proprietà | Valore |
|-----------|--------|
| File | `agricultural_field_path_planning_dataset.csv` |
| Tipologie di campo | Regolari, irregolari, con ostacoli |
| Feature | Descrittori geometrici + metriche di traiettoria |
| Target di regressione | Orientamento ottimale (`label_orientation_x`, `label_orientation_y`) |
| Target di classificazione | Tipologia del campo (`field_type`) |
| Preprocessing | StandardScaler (scikit-learn) |

---

## 🚀 Avvio rapido

### 1. Installare le dipendenze

```bash
pip install -r requirements.txt
```

Librerie principali: `PyTorch`, `NumPy`, `Pandas`, `Shapely`, `Matplotlib`, `scikit-learn`.

### 2. Generare il dataset

```bash
python src/generation/dataset_generator.py
```

Questo comando genera `data/agricultural_field_path_planning_dataset.csv`.

### 3. Addestrare FieldNet

```bash
python src/ml/rete_neurale_predittiva.py
```

Il modello addestrato viene salvato come `fieldnet_multitask.pt`.

### 4. Visualizzare i risultati

```bash
python src/viz/visualizza_csv.py
python src/viz/genera_grafico.py
```

---

## 🔑 Algoritmi principali

| Modulo | Algoritmo | Descrizione |
|--------|-----------|-------------|
| `decomposizione_area.py` | Decomposizione trapezoidale | Suddivide l’area di lavoro in sotto-celle convexe tramite eventi di scansione |
| `genera_capezzagne_bordocampo.py` | HeadlandCalculator | Calcola le strisce interne di bordo per le manovre di fine campo |
| `genera_capezzagne_ostacoli.py` | Capezzagne sugli ostacoli | Genera buffer concentrici attorno agli ostacoli statici |
| `genera_traiettoria_sottocella.py` | Boustrophedon + Bézier | Genera passaggi paralleli con curve di svolta morbide, rispettando il raggio di sterzata del veicolo |
| `rete_neurale_predittiva.py` | FieldNet (DNN multi-task) | Classifica la tipologia del campo e stima l’orientamento ottimale |

---

## 📈 Risultati

FieldNet è stata addestrata e valutata sul dataset generato automaticamente.

- **Task di classificazione**: predizione della tipologia del campo (regolare / irregolare / con ostacoli).
- **Task di regressione**: predizione del vettore di orientamento ottimale (X, Y).
- L’addestramento è stato monitorato per **100 epoche**, tracciando loss di training e validation.
- I pesi finali del modello sono salvati in `fieldnet_multitask.pt`.

---

## 🔭 Sviluppi futuri

- Integrazione con dati reali di posizionamento GNSS/RTK.
- Gestione di ostacoli dinamici (persone, mezzi in movimento).
- Estensione a modelli di terreno 3D (DEM/DTM).
- Ricalcolo in tempo reale del percorso su copertura parziale del campo.

---

## 📄 Licenza

Progetto accademico — Università della Calabria, A.A. 2024/2025.  
Uso consentito per finalità di studio e ricerca.