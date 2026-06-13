import pandas as pd

# Percorso del tuo file CSV
csv_file_path = "agricultural_field_path_planning_dataset.csv"

try:
    df = pd.read_csv(csv_file_path)
    print(f"Dataset caricato con successo. Numero totale di righe: {len(df)}")
    print("\nIntestazione del dataset:")
    print(df.head())

    # Assicurati che la colonna 'field_type' esista
    if 'field_type' not in df.columns:
        print("\nErrore: La colonna 'field_type' non è presente nel dataset. Impossibile raggruppare per tipo di campo.")
        # Se non c'è 'field_type', potremmo provare a dedurre dal numero di riga,
        # ma è meglio avere la colonna esplicita come previsto dal `dataset_generator.py`
        # Se sai che le prime 100 sono di tipo A, le seconde 100 di tipo B, ecc.,
        # puoi creare una colonna 'field_type' basata sull'indice:
        # df['field_type'] = 'unknown'
        # df.loc[df.index < 100, 'field_type'] = 'regolare_senza_ostacoli'
        # df.loc[(df.index >= 100) & (df.index < 200), 'field_type'] = 'irregolare_senza_ostacoli'
        # df.loc[df.index >= 200, 'field_type'] = 'irregolare_con_ostacoli'
        exit() # Termina lo script se la colonna non c'è

    print("\n--- Analisi per tipo di campo ---")

    # Ottieni i tipi unici di campo presenti nel dataset
    field_types = df['field_type'].unique()
    print(f"Tipi di campo trovati: {field_types}")

    for field_type in field_types:
        print(f"\n======== Analisi per: '{field_type}' ========")
        subset_df = df[df['field_type'] == field_type].copy() # .copy() per evitare SettingWithCopyWarning
        
        if subset_df.empty:
            print(f"Nessun dato per il tipo di campo '{field_type}'.")
            continue
        
        print(f"Numero di campi di tipo '{field_type}': {len(subset_df)}")

        # Statistiche descrittive generali per le colonne numeriche
        print("\nStatistiche Descrittive (Numeriche):")
        # Seleziona solo le colonne numeriche rilevanti per l'analisi e ignora le colonne di coordinate
        numerical_cols = [
            'original_area_area', 'original_area_perimeter', 'original_area_width', 'original_area_height',
            'original_area_aspect_ratio', 'num_vertices_area', 'num_obstacles', 'total_obstacle_area',
            'obstacle_area_ratio', 'area_capezzagne_ostacoli', 'area_capezzagne_bordocampo',
            'num_decomposed_cells', 'total_trajectory_length',
            'label_orientation_x', 'label_orientation_y'
        ]
        # Filtra le colonne che effettivamente esistono nel subset_df
        existing_numerical_cols = [col for col in numerical_cols if col in subset_df.columns]
        print(subset_df[existing_numerical_cols].describe().to_string()) # .to_string() per una migliore formattazione nell'output testuale

        # Conteggi per le colonne booleane
        print("\nConteggi per flag booleane:")
        boolean_cols = ['is_irregular_field', 'is_irregular_shape', 'has_obstacles']
        for col in boolean_cols:
            if col in subset_df.columns:
                print(f"- {col}:\n{subset_df[col].value_counts().to_string()}")

        # Verifica di coerenza specifica per ogni tipo di campo
        if field_type == 'regolare_senza_ostacoli':
            print("\nVerifiche specifiche per 'regolare_senza_ostacoli':")
            # is_irregular_shape dovrebbe essere prevalentemente False
            print(f"  'is_irregular_shape' dovrebbe essere False: {subset_df['is_irregular_shape'].value_counts(normalize=True).get(False, 0):.2f}")
            # has_obstacles dovrebbe essere False
            print(f"  'has_obstacles' dovrebbe essere False: {subset_df['has_obstacles'].value_counts(normalize=True).get(False, 0):.2f}")
            # num_obstacles dovrebbe essere 0
            print(f"  'num_obstacles' media (dovrebbe essere vicino a 0): {subset_df['num_obstacles'].mean():.2f}")
            print(f"  'total_obstacle_area' media (dovrebbe essere vicino a 0): {subset_df['total_obstacle_area'].mean():.2f}")
            print(f"  'area_capezzagne_ostacoli' media (dovrebbe essere vicino a 0): {subset_df['area_capezzagne_ostacoli'].mean():.2f}")


        elif field_type == 'irregolare_senza_ostacoli':
            print("\nVerifiche specifiche per 'irregolare_senza_ostacoli':")
            # is_irregular_shape dovrebbe essere prevalentemente True
            print(f"  'is_irregular_shape' dovrebbe essere True: {subset_df['is_irregular_shape'].value_counts(normalize=True).get(True, 0):.2f}")
            # has_obstacles dovrebbe essere False
            print(f"  'has_obstacles' dovrebbe essere False: {subset_df['has_obstacles'].value_counts(normalize=True).get(False, 0):.2f}")
            # num_obstacles dovrebbe essere 0
            print(f"  'num_obstacles' media (dovrebbe essere vicino a 0): {subset_df['num_obstacles'].mean():.2f}")
            print(f"  'total_obstacle_area' media (dovrebbe essere vicino a 0): {subset_df['total_obstacle_area'].mean():.2f}")
            print(f"  'area_capezzagne_ostacoli' media (dovrebbe essere vicino a 0): {subset_df['area_capezzagne_ostacoli'].mean():.2f}")


        elif field_type == 'irregolare_con_ostacoli':
            print("\nVerifiche specifiche per 'irregolare_con_ostacoli':")
            # is_irregular_shape dovrebbe essere prevalentemente True
            print(f"  'is_irregular_shape' dovrebbe essere True: {subset_df['is_irregular_shape'].value_counts(normalize=True).get(True, 0):.2f}")
            # has_obstacles dovrebbe essere True
            print(f"  'has_obstacles' dovrebbe essere True: {subset_df['has_obstacles'].value_counts(normalize=True).get(True, 0):.2f}")
            # num_obstacles dovrebbe essere > 0
            print(f"  'num_obstacles' media (dovrebbe essere > 0): {subset_df['num_obstacles'].mean():.2f}")
            print(f"  'total_obstacle_area' media (dovrebbe essere > 0): {subset_df['total_obstacle_area'].mean():.2f}")
            print(f"  'obstacle_area_ratio' media (dovrebbe essere > 0): {subset_df['obstacle_area_ratio'].mean():.2f}")
            print(f"  'area_capezzagne_ostacoli' media (dovrebbe essere > 0): {subset_df['area_capezzagne_ostacoli'].mean():.2f}")
            
except FileNotFoundError:
    print(f"Errore: Il file '{csv_file_path}' non è stato trovato. Assicurati che si trovi nella stessa directory dello script o specifica il percorso completo.")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura o l'analisi del CSV: {e}")