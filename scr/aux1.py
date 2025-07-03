import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset_70_30(save_folder):
    """
    Esta función realiza la separación aleatoria 70% entrenamiento y 30% validación
    de los archivos X.npy, Y.npy y dataset_variables.csv ya generados en la carpeta especificada.
    Guarda automáticamente los archivos separados en la misma carpeta.

    Ingresa:
        save_folder (str): ruta donde están almacenados X.npy, Y.npy y dataset_variables.csv
    Retorna:
        None
    """

    # Cargar datos completos previamente generados
    X = np.load(os.path.join(save_folder, 'X.npy'))
    Y = np.load(os.path.join(save_folder, 'Y.npy'))
    registros = pd.read_csv(os.path.join(save_folder, 'dataset_variables.csv'), sep=';')

    # Separación reproducible 70% entrenamiento y 30% validación
    X_train, X_val, Y_train, Y_val, registros_train, registros_val = train_test_split(
        X, Y, registros, test_size=0.3, random_state=42, shuffle=True
    )

    # Guardar datos de entrenamiento
    np.save(os.path.join(save_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(save_folder, 'Y_train.npy'), Y_train)
    registros_train.to_csv(os.path.join(save_folder, 'dataset_variables_train.csv'), index=False, sep=';')

    # Guardar datos de validación
    np.save(os.path.join(save_folder, 'X_val.npy'), X_val)
    np.save(os.path.join(save_folder, 'Y_val.npy'), Y_val)
    registros_val.to_csv(os.path.join(save_folder, 'dataset_variables_val.csv'), index=False, sep=';')

    print(f"Separación completada: entrenamiento (70%) y validación (30%) en {save_folder}.")
