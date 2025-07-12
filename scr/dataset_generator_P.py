from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = Path().resolve()
sys.path.append(str(BASE_DIR.parent / 'scr'))

from solver_fd import temp_chapa_P
from dataset_generator import variables  # Usa tu función "variables" existente

def generar_muestra(args):
    Nx, Ny, dx, dy = args

    cond_contor, typ_cond_contorno, hot_point, k, material_nombre, T_fusion = variables(Nx, Ny)
    tipo_map = {'temp': 0, 'flu': 1}

    tipo_A = tipo_map[typ_cond_contorno['A']]
    tipo_B = tipo_map[typ_cond_contorno['B']]
    tipo_C = tipo_map[typ_cond_contorno['C']]
    tipo_D = tipo_map[typ_cond_contorno['D']]

    x_muestra = [
        k,
        hot_point['T'],
        hot_point['i'],
        hot_point['j'],
        tipo_A,
        tipo_B,
        tipo_C,
        tipo_D,
        cond_contor['A'],
        cond_contor['B'],
        cond_contor['C'],
        cond_contor['D']
    ]

    T = temp_chapa_P(cond_contor, Nx, Ny, typ_cond_contorno, dx, dy, k, hot_point)
    y_muestra = T.flatten()

    registro = {
        'material': material_nombre,
        'k': k,
        'T_fusion': T_fusion,
        'T_hp': hot_point['T'],
        'i_hp': hot_point['i'],
        'j_hp': hot_point['j'],
        'tipo_A': typ_cond_contorno['A'],
        'tipo_B': typ_cond_contorno['B'],
        'tipo_C': typ_cond_contorno['C'],
        'tipo_D': typ_cond_contorno['D'],
        'valor_A': cond_contor['A'],
        'valor_B': cond_contor['B'],
        'valor_C': cond_contor['C'],
        'valor_D': cond_contor['D']
    }

    return x_muestra, y_muestra, registro

def generar_dataset_paralelo(n_muestras, Nx, Ny, dx, dy, subfolder_name=None):

    #........................................................................................................................
    #........................................................................................................................

    #   Se definen las listas donde se almacenaran las muestras y los registros
    X, Y, registros = [], [], []

    #   ................... REVISAR ............................................
    args_list = [(Nx, Ny, dx, dy) for _ in range(n_muestras)]


    #   ................... REVISAR ............................................
    with ProcessPoolExecutor() as executor:

        futures = [executor.submit(generar_muestra, args) for args in args_list]

        for idx, future in enumerate(tqdm(as_completed(futures), total=n_muestras, desc="Generando muestras")):

            try:

                x_muestra, y_muestra, registro = future.result()
                X.append(x_muestra)
                Y.append(y_muestra)
                registros.append(registro)

            except Exception as e:
                print(f"Error en la muestra {idx}: {e}")

    #........................................................................................................................
    #........................................................................................................................

    X = np.array(X)
    Y = np.array(Y)

    df_registros = pd.DataFrame(registros)

    T_min = Y.min(axis=1)
    T_max = Y.max(axis=1)
    T_fusion = df_registros['T_fusion'].values

    #idx_validos = (T_min >= -100000) & (T_max <= (T_fusion + 100000))
    idx_validos = (T_min >= -250) & (T_max <= (T_fusion - 10))

    X = X[idx_validos]
    Y = Y[idx_validos]
    df_registros = df_registros.iloc[idx_validos].reset_index(drop=True)

    print(f"\nFiltrado completado:")
    print(f"Muestras originales: {n_muestras}")
    print(f"Muestras después del filtrado: {len(Y)}")

    base_folder = BASE_DIR.parent / 'data'

    if subfolder_name is not None:
        save_folder = os.path.join(base_folder, subfolder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f'Se creó la carpeta: {save_folder}')
    else:
        save_folder = base_folder

    X_train, X_val, Y_train, Y_val, registros_train, registros_val = train_test_split(
        X, Y, df_registros, test_size=0.3, random_state=42, shuffle=True)

    np.save(os.path.join(save_folder, 'X.npy'), X)
    np.save(os.path.join(save_folder, 'Y.npy'), Y)
    df_registros.to_csv(os.path.join(save_folder, 'dataset_variables.csv'), index=False, sep=';')

    np.save(os.path.join(save_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(save_folder, 'Y_train.npy'), Y_train)
    registros_train.to_csv(os.path.join(save_folder, 'dataset_variables_train.csv'), index=False, sep=';')

    np.save(os.path.join(save_folder, 'X_val.npy'), X_val)
    np.save(os.path.join(save_folder, 'Y_val.npy'), Y_val)
    registros_val.to_csv(os.path.join(save_folder, 'dataset_variables_val.csv'), index=False, sep=';')

    #print(f"Dataset generado y guardado en {save_folder} con separación entrenamiento/validación.")
    return len(Y)

if __name__ == "__main__":
    generar_dataset_paralelo(
        n_muestras=1000,
        Nx=50,
        Ny=50,
        dx=0.05,
        dy=0.05,
        subfolder_name="Dataset de 1000 muestras con filtro"
    )
