from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from IPython.display import display

from utils import graficarChapa, comparar_T
from mlp_temp_regressor import MLPTempRegressor

def verificar_modelo(folder_data, folder_results, idx_muestra=None):

    BASE_DIR = Path().resolve()

    results_path = BASE_DIR.parent /'results'/ folder_results
    data_path = BASE_DIR.parent / 'data' / folder_data

    # ------------------ Cargar normalizadores ------------------

    mean_X = np.load(results_path / 'mean_X.npy')
    std_X = np.load(results_path / 'std_X.npy')
    mean_Y = np.load(results_path / 'mean_Y.npy').item()
    std_Y = np.load(results_path / 'std_Y.npy').item()

    # ------------------ Cargar datos ------------------

    X_data = np.load(data_path / 'X_train.npy').astype(np.float32)
    Y_data = np.load(data_path / 'Y_train.npy').astype(np.float32)

    # ------------------ Normalizar entrada ------------------

    X_data_norm = (X_data - mean_X) / std_X

    input_dim = X_data.shape[1]
    output_dim = Y_data.shape[1]

    # ------------------ Cargar modelo ------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLPTempRegressor(input_dim, output_dim).to(device)

    model_save_path = results_path / 'model_train.pt'

    #print(model_save_path)

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    #print("Modelo cargado correctamente.")
    
    # ------------------ Inferencia ------------------

    if idx_muestra is None:
        idx_muestra = np.random.randint(len(X_data))

    X_sample_tensor = torch.tensor(X_data_norm[idx_muestra]).unsqueeze(0).to(device)
    Y_pred_norm = model(X_sample_tensor).detach().cpu().numpy().squeeze()

    # ------------------ Desnormalización ------------------

    Y_pred = Y_pred_norm * std_Y + mean_Y

    # ------------------ Preparar para graficar ------------------

    Y_pred_img = Y_pred.reshape(50, 50)
    Y_true_img = Y_data[idx_muestra].reshape(50, 50)

    #................... Tablas comparativas .....................

    valores = {
        "Métrica": ["Mínimo (°C)", "Máximo (°C)", "Promedio (°C)", "Desvío estándar"],
        "Valor Real": [
            np.round(Y_true_img.min(),2),
            np.round(Y_true_img.max(),2),
            np.round(Y_true_img.mean(),2),
            np.round(Y_true_img.std(),2)
        ],
        "Predicción ML": [
            np.round(Y_pred.min(),2),
            np.round(Y_pred.max(),2),
            np.round(Y_pred.mean(),2),
            np.round(Y_pred.std(),2)
        ]
    }

    df_resultados = pd.DataFrame(valores)

    print("\nTabla comparativa de resultados:")
    display(df_resultados)

    # ------------------ Visualización ------------------
    
    comparar_T(Y_true_img, Y_pred_img, 50, 50, etiquetas=('Diferencias Finitas', 'Predicción ML'))




