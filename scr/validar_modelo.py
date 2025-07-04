import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mlp_temp_regressor import MLPTempRegressor
from utils import comparar_T

def validar_modelo(folder_data,folder_results,idx_muestra=None,plot_hist=True):

    BASE_DIR = Path().resolve()

    results_path = BASE_DIR.parent / 'results'/ folder_results
    data_path = BASE_DIR.parent / 'data' / folder_data

    #print(results_path)

    # ------------------ Cargar normalizadores ------------------

    mean_X = np.load(results_path / 'mean_X.npy')
    std_X = np.load(results_path / 'std_X.npy')
    mean_Y = np.load(results_path / 'mean_Y.npy').item()
    std_Y = np.load(results_path / 'std_Y.npy').item()

    # ------------------ Cargar datos ------------------

    X_data = np.load(data_path / 'X_val.npy').astype(np.float32)
    Y_data = np.load(data_path / 'Y_val.npy').astype(np.float32)

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

    # ------------------------------------------------------------------------
    # ---------------- Predicciones ------------------------------------------

    with torch.no_grad():
        X_tensor = torch.tensor(X_data_norm).to(device)
        Y_pred_norm = model(X_tensor).cpu().numpy()

    # ---------------- Desnormalización ----------------

    Y_pred = Y_pred_norm * std_Y + mean_Y

    # ---------------- Métricas ----------------

    mse = np.mean((Y_pred - Y_data) ** 2)
    mae = np.mean(np.abs(Y_pred - Y_data))

    print(f"\nResultados en conjunto de validación:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")

    # ---------------- Visualización muestra ----------------

    if idx_muestra is None:
        idx_muestra = np.random.randint(len(X_data))

    Y_pred_img = Y_pred[idx_muestra].reshape(50,50)
    Y_true_img = Y_data[idx_muestra].reshape(50,50)

    print(f"\nMostrando muestra índice: {idx_muestra}")
    comparar_T(Y_true_img, Y_pred_img, 50, 50, etiquetas=('Valor Real', 'Predicción ML'))

    # ---------------- Histograma de errores ----------------
    if plot_hist:

        errores = (Y_pred - Y_data).flatten()
        plt.figure(figsize=(8, 4))
        plt.hist(errores, bins=100, alpha=0.7, color='steelblue')
        plt.title("Distribución de errores en validación")
        plt.xlabel("Error (°C)")
        plt.ylabel("Frecuencia")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
