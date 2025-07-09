import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import pandas as pd
from pathlib import Path

import torch
import numpy as np
from mlp_temp_regressor import MLPTempRegressor
from solver_fd import temp_chapa_P,temp_chapa_P2

def graficarChapa(T,Nx,Ny):

    fig, ax = plt.subplots(figsize=(8, 6))

    numSeparaciones = 100  
    numLineas = 16     

    mappable = ax.contourf(T.reshape(Nx, Ny), levels=numSeparaciones, origin='upper', cmap='plasma')

    levels = ax.contour(T.reshape(Nx, Ny), levels=numLineas, colors='k', linewidths=1,origin='upper')

    cbar = plt.colorbar(mappable)
    cbar.set_label('T (°C)')

    ax.clabel(levels, inline=True, fontsize=10, fmt='%1.0f')


    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_title('Distribución de Temperatura para una placa cuadrada')

    plt.tight_layout()
    plt.show()

    return

#....................................................................................................................


def mostrar_tabla_variables_ordenada(cond_contor, typ_cond_contorno, hot_point, material_nombre):

    print("\n=== Condiciones de Contorno ===")

    tabla_contorno = []
    for borde in cond_contor.keys():
        tipo = typ_cond_contorno.get(borde, 'No definido')
        valor = cond_contor[borde]
        tabla_contorno.append([borde, tipo, valor])

    print(tabulate(tabla_contorno, headers=["Borde", "Tipo de condición", "Valor"], tablefmt="grid"))

    print("\n=== Punto Caliente ===")

    tabla_hot = []

    tabla_hot = [[hot_point['i'], hot_point['j'], hot_point['T']]]
    print(tabulate(tabla_hot, headers=["i", "j", "Temp"], tablefmt="grid"))

    print(f"\nMaterial: {material_nombre}")

def comparar_T(T1, T2, Nx, Ny, folder,etiquetas=('Versión 1', 'Versión 2'),escala=True):
    """
    Compara dos distribuciones de temperatura T1 y T2:
    - Muestra las dos distribuciones lado a lado.
    - Muestra la diferencia absoluta entre ellas.
    - Calcula métricas de diferencia.

    Parámetros:
        T1, T2: arrays 1D o 2D de temperatura.
        Nx, Ny: dimensiones de la malla.
        etiquetas: tupla con etiquetas para las versiones.
    """

    # Asegura que tengan forma (Ny, Nx)
    T1 = np.array(T1).reshape((Ny, Nx))
    T2 = np.array(T2).reshape((Ny, Nx))
    diferencia = np.abs(T1 - T2)

    # Métricas de diferencia
    diff_max = np.max(diferencia)
    diff_mean = np.mean(diferencia)
    diff_std = np.std(diferencia)

    print(f"\nComparación de distribuciones:")
    print(f"Diferencia máxima: {diff_max:.4f} °C")
    print(f"Diferencia media:  {diff_mean:.4f} °C")
    print(f"Desvío estándar:   {diff_std:.4f} °C")

    
    # Determinar escala común
    vmin = min(np.min(T1), np.min(T2))
    vmax = max(np.max(T1), np.max(T2))

    # Gráficos
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    if escala:
        im0 = axs[0].imshow(T1, origin='lower', cmap='plasma',vmin=vmin,vmax=vmax)
        axs[0].set_title(f'{etiquetas[0]}')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(T2, origin='lower', cmap='plasma',vmin=vmin,vmax=vmax)
        axs[1].set_title(f'{etiquetas[1]}')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    else:
        im0 = axs[0].imshow(T1, origin='lower', cmap='plasma')
        axs[0].set_title(f'{etiquetas[0]}')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(T2, origin='lower', cmap='plasma')
        axs[1].set_title(f'{etiquetas[1]}')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(diferencia, origin='lower', cmap='viridis')
    axs[2].set_title('Diferencia absoluta |T1 - T2| [°C]')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel('i (x)')
        ax.set_ylabel('j (y)')

    plt.suptitle(f'Comparación de distribuciones de temperatura - {folder}')
    plt.tight_layout()
    plt.show()


def graficar_hist_k():

    BASE_DIR = Path().resolve()

    csv_path = BASE_DIR.parent / 'data' / 'materiales.csv'

    df_materiales = pd.read_csv(csv_path, sep=';')

    print("Columnas disponibles en el CSV:")
    print(df_materiales.columns)

    columna_k = 'k [W/m·K]'

    plt.figure(figsize=(10,5))
    plt.hist(df_materiales[columna_k], bins=20, edgecolor='black')
    plt.title("Distribución de la conductividad térmica (k) en materiales.csv")
    plt.xlabel("k [W/m·K]")
    plt.ylabel("Cantidad de materiales")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Valor mínimo de k: {df_materiales[columna_k].min():.2f} W/m·K")
    print(f"Valor máximo de k: {df_materiales[columna_k].max():.2f} W/m·K")
    print(f"Valor medio de k: {df_materiales[columna_k].mean():.2f} W/m·K")
    print(f"Desvío estándar de k: {df_materiales[columna_k].std():.2f} W/m·K")



def compararLoss(subfolder_name_1,subfolder_name_2,aux):

    BASE_DIR = Path().resolve()
    save_folder_1 = BASE_DIR.parent / 'results' / subfolder_name_1
    save_folder_2 = BASE_DIR.parent / 'results' / subfolder_name_2

    loss_train_1 = np.load(save_folder_1 / 'loss_history_Train.npy')
    loss_val_1 = np.load(save_folder_1 / 'loss_history_Val.npy')

    loss_train_2 = np.load(save_folder_2 / 'loss_history_Train.npy')
    loss_val_2 = np.load(save_folder_2 / 'loss_history_Val.npy')

    plt.figure(figsize=(10, 6))

    if aux == 0 or aux == 1:
        plt.semilogy(loss_train_1, label=f'{subfolder_name_1} - Train', linewidth=1, linestyle='--', marker = 'o', markersize = 2)
        plt.semilogy(loss_val_1, label=f'{subfolder_name_1} - Val', linewidth=1, linestyle='--', marker = 'o', markersize = 2)

    if aux == 0 or aux == 2:
        plt.semilogy(loss_train_2, label=f'{subfolder_name_2} - Train', linewidth=1, linestyle='--', marker = 'o', markersize = 2)
        plt.semilogy(loss_val_2, label=f'{subfolder_name_2} - Val', linewidth=1, linestyle='--', marker = 'o', markersize = 2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Comparación de Curvas de Pérdida de Modelos')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

def compararLossM(subfolder_list , aux, ylimits=None, xlimits=None , radio = True):

    BASE_DIR = Path().resolve()
    plt.figure(figsize=(10, 6))

    for subfolder_name in subfolder_list:
        save_folder = BASE_DIR.parent / 'results' / subfolder_name

        loss_train = np.load(save_folder / 'loss_history_Train.npy')
        loss_val = np.load(save_folder / 'loss_history_Val.npy')

        if radio: 
            
            loss_radio_train = np.load(save_folder / 'loss_radio_history_Train.npy')
            loss_radio_val = np.load(save_folder / 'loss_radio_history_Val.npy')

            loss_resto_train = np.load(save_folder / 'loss_resto_history_Train.npy')
            loss_resto_val = np.load(save_folder / 'loss_resto_history_Val.npy')


        epochs = np.arange(len(loss_train))

        if aux == 0 or aux == 1:
            plt.semilogy(epochs, loss_train,label=f'{subfolder_name} - Train',linewidth=1, linestyle='--',marker='o', markersize=2)

        if aux == 0 or aux == 2:
            plt.semilogy(epochs, loss_val,label=f'{subfolder_name} - Val',linewidth=1, linestyle='--',marker='o', markersize=2)

        if aux == 3 or aux == 4:
            plt.semilogy(epochs, loss_resto_train,label=f'{subfolder_name} - Resto - Train',linewidth=1, linestyle='--',marker='o', markersize=2)
            plt.semilogy(epochs, loss_radio_train,label=f'{subfolder_name} - RADIO - Train',linewidth=1, linestyle='--',marker='o', markersize=2)

        if aux == 3 or aux == 5:
            plt.semilogy(epochs, loss_resto_val,label=f'{subfolder_name} - Resto - Val',linewidth=1, linestyle='--',marker='o', markersize=2)
            plt.semilogy(epochs, loss_radio_val,label=f'{subfolder_name} - RADIO - Val',linewidth=1, linestyle='--',marker='o', markersize=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Comparación de las curvas de pérdida')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    if ylimits is not None:
        plt.ylim(ylimits)
    
    if xlimits is not None:
        plt.xlim(xlimits)

    plt.tight_layout()
    plt.show()

#.........................................................................................................

def graficar_tiempos(csv_path):

    df = pd.read_csv(csv_path)

    # Filtrar solo los casos con muestras válidas (evita errores si hubo n_validas = 0)
    df = df[(df["n_filtrado_sec"] > 0) & (df["n_filtrado_par"] > 0)]

    plt.figure(figsize=(8,5))
    plt.plot(df["n_muestras"], df["tiempo_sec_s"]/60, marker='o', label="Secuencial")
    plt.plot(df["n_muestras"], df["tiempo_par_s"]/60, marker='o', label="Paralelo")
    plt.xlabel("Cantidad de muestras generadas")
    plt.ylabel("Tiempo de ejecución [min]")
    plt.title("Tiempo de ejecución: Secuencial vs Paralelo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#................................................................................................

import time

def validar_modelo(folder_data,folder_results,dato,idx_muestra=None,mostrar=True,escala=True):

    BASE_DIR = Path().resolve()

    results_path = BASE_DIR.parent / 'results'/ folder_results
    data_path = BASE_DIR.parent / 'data' / folder_data

    # ------------------ Cargar datos ------------------

    if dato == 'val':

        X_data = np.load(data_path / 'X_val.npy').astype(np.float32)
        Y_data = np.load(data_path / 'Y_val.npy').astype(np.float32)
        df_variables = pd.read_csv(data_path / 'dataset_variables_val.csv', sep=';')

        mean_X = np.load(results_path / 'val_mean_X.npy')
        std_X = np.load(results_path / 'val_std_X.npy')
        mean_Y = np.load(results_path / 'val_mean_Y.npy').item()
        std_Y = np.load(results_path / 'val_std_Y.npy').item()

    elif dato == 'train':
                
        X_data = np.load(data_path / 'X_train.npy').astype(np.float32)
        Y_data = np.load(data_path / 'Y_train.npy').astype(np.float32)
        df_variables = pd.read_csv(data_path / 'dataset_variables_train.csv', sep=';')

        mean_X = np.load(results_path / 'mean_X.npy')
        std_X = np.load(results_path / 'std_X.npy')
        mean_Y = np.load(results_path / 'mean_Y.npy').item()
        std_Y = np.load(results_path / 'std_Y.npy').item()

    if idx_muestra is None:
        idx_muestra = np.random.randint(len(X_data))

    # ------------ CONSTRUIR TABLA --------------------------------------------------

    registro = df_variables.iloc[idx_muestra]

    cond_contor = {
        'A': registro['valor_A'],
        'B': registro['valor_B'],
        'C': registro['valor_C'],
        'D': registro['valor_D']
    }

    typ_cond_contorno = {
        'A': registro['tipo_A'],
        'B': registro['tipo_B'],
        'C': registro['tipo_C'],
        'D': registro['tipo_D']
    }

    hot_point = {
        'T': registro['T_hp'],
        'i': int(registro['i_hp']),
        'j': int(registro['j_hp'])
    }

    material_nombre = registro['material']

    # -------------------CARGAR MODELO --------------------------------

    input_dim = X_data.shape[1]
    output_dim = Y_data.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLPTempRegressor(input_dim, output_dim).to(device)

    model_save_path = results_path / 'model_train.pt'

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    #----------------------------------------------------------------------------

    X_muestra = X_data[idx_muestra]
    Y_true = Y_data[idx_muestra]

    X_muestra_norm = (X_muestra - mean_X) / std_X

    X_tensor = torch.tensor(X_muestra_norm).to(device)

    start_time = time.perf_counter()

    with torch.no_grad():   # Desactiva el calculo de gradientes ... "PyTorch construye el grafo de cómputo y almacena todos los tensores intermedios"
        Y_pred_norm = model(X_tensor).cpu().numpy()

    end_time = time.perf_counter()
    tiempo_total = end_time - start_time

    print(f"Tiempo de calculo con ML: {tiempo_total:.6f} segundos")

    # ----------------------------------------------------------------------------------

    start_time_1 = time.perf_counter()
    T = temp_chapa_P(cond_contor, 50, 50, typ_cond_contorno, 0.05, 0.05, registro['k'], hot_point)
    end_time_1 = time.perf_counter()
    tiempo_fd_1 = end_time_1 - start_time_1

    start_time_2 = time.perf_counter()
    T = temp_chapa_P2(cond_contor, 50, 50, typ_cond_contorno, 0.05, 0.05, registro['k'], hot_point)
    end_time_2 = time.perf_counter()
    tiempo_fd_2 = end_time_2 - start_time_2

    print(f"Tiempo de cálculo con MDF_1 (temp_chapa_P): {tiempo_fd_1:.6f} segundos")
    print(f"Tiempo de cálculo con MDF_2 (temp_chapa_P): {tiempo_fd_2:.6f} segundos")

    # ---------------- Desnormalización ----------------

    Y_pred = Y_pred_norm * std_Y + mean_Y

    # ---------------- Visualización muestra ----------------

    Y_pred_img = Y_pred.reshape(50, 50)
    Y_true_img = Y_true.reshape(50, 50)

    print(f"\nSe muestra la muestra con índice: {idx_muestra}")

    if mostrar:
        mostrar_tabla_variables_ordenada(cond_contor, typ_cond_contorno, hot_point, material_nombre)


    comparar_T(Y_true_img, Y_pred_img, 50, 50,folder_results ,etiquetas=('Valor Real', 'Predicción ML'),escala=escala)

def comparar_MT(T1, T2,T_true, folder,etiquetas=('Versión 1', 'Versión 2'),auto=True):

    # Asegura que tengan forma (Ny, Nx)
    T1 = np.array(T1).reshape((50, 50))
    T2 = np.array(T2).reshape((50, 50))
    T_true = np.array(T_true).reshape((50,50))

    diferencia1 = np.abs(T1 - T_true)
    diferencia2 = np.abs(T2 - T_true)

    # Métricas de diferencia

    diff_max1 = np.max(diferencia1)
    diff_mean1 = np.mean(diferencia1)
    diff_std1 = np.std(diferencia1)

    diff_max2 = np.max(diferencia2)
    diff_mean2 = np.mean(diferencia2)
    diff_std2 = np.std(diferencia2)

    print(f"\nComparación de distribuciones - {etiquetas[0]}")
    print(f"Diferencia máxima: {diff_max1:.4f} °C  -  Diferencia media:  {diff_mean1:.4f} °C  -  Desvío estándar:   {diff_std1:.4f} °C")

    print(f"\nComparación de distribuciones - {etiquetas[1]}")
    print(f"Diferencia máxima: {diff_max2:.4f} °C  -  Diferencia media:  {diff_mean2:.4f} °C  -  Desvío estándar:   {diff_std2:.4f} °C")
    
    if auto:
        # Determinar escala común
        vmin = min(np.min(T1), np.min(T2),np.min(T_true))
        vmax = max(np.max(T1), np.max(T2),np.max(T_true))

        # Gráficos
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))

        im0 = axs[0].imshow(T_true, origin='lower', cmap='plasma',vmin=vmin,vmax=vmax)
        axs[0].set_title('Muestra original')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(T1, origin='lower', cmap='plasma',vmin=vmin,vmax=vmax)
        axs[1].set_title(f'{etiquetas[0]}')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(T2, origin='lower', cmap='plasma',vmin=vmin,vmax=vmax)
        axs[2].set_title(f'{etiquetas[1]}')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    else:
        
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))

        im0 = axs[0].imshow(T_true, origin='lower', cmap='plasma')
        axs[0].set_title('Muestra original')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(T1, origin='lower', cmap='plasma')
        axs[1].set_title(f'{etiquetas[0]}')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(T2, origin='lower', cmap='plasma')
        axs[2].set_title(f'{etiquetas[1]}')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel('i (x)')
        ax.set_ylabel('j (y)')

    plt.suptitle(f'Comparación de distribuciones de temperatura - {folder}')
    plt.tight_layout()
    plt.show()

def compararModelos(folder_data,folder_resultsM,dato,idx_muestra=None,mostrar=True,auto=True):

    BASE_DIR = Path().resolve()
    data_path = BASE_DIR.parent / 'data' / folder_data

    if dato == 'val':
        X_data = np.load(data_path / 'X_val.npy').astype(np.float32)
        Y_data = np.load(data_path / 'Y_val.npy').astype(np.float32)
        df_variables = pd.read_csv(data_path / 'dataset_variables_val.csv', sep=';')
    elif dato == 'train':         
        X_data = np.load(data_path / 'X_train.npy').astype(np.float32)
        Y_data = np.load(data_path / 'Y_train.npy').astype(np.float32)
        df_variables = pd.read_csv(data_path / 'dataset_variables_train.csv', sep=';')
 

    y_results = []
    for folder_results in folder_resultsM:
        results_path = BASE_DIR.parent / 'results'/ folder_results
        if dato == 'val':
            mean_X = np.load(results_path / 'val_mean_X.npy')
            std_X = np.load(results_path / 'val_std_X.npy')
            mean_Y = np.load(results_path / 'val_mean_Y.npy').item()
            std_Y = np.load(results_path / 'val_std_Y.npy').item()
        elif dato == 'train':         
            mean_X = np.load(results_path / 'mean_X.npy')
            std_X = np.load(results_path / 'std_X.npy')
            mean_Y = np.load(results_path / 'mean_Y.npy').item()
            std_Y = np.load(results_path / 'std_Y.npy').item()

        if idx_muestra is None:
            idx_muestra = np.random.randint(len(X_data))

        input_dim = X_data.shape[1]
        output_dim = Y_data.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLPTempRegressor(input_dim, output_dim).to(device)
        model_save_path = results_path / 'model_train.pt'
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        X_muestra = X_data[idx_muestra]
        X_muestra_norm = (X_muestra - mean_X) / std_X
        X_tensor = torch.tensor(X_muestra_norm).to(device)
        with torch.no_grad():   # Desactiva el calculo de gradientes ... "PyTorch construye el grafo de cómputo y almacena todos los tensores intermedios"
            Y_pred_norm = model(X_tensor).cpu().numpy()
        Y_pred = Y_pred_norm * std_Y + mean_Y
        y_results.append(Y_pred.reshape(50, 50))

    Y_true = Y_data[idx_muestra]
    Y_true_img = Y_true.reshape(50, 50)

    print(f"\nSe muestra la muestra con índice: {idx_muestra}")
    registro = df_variables.iloc[idx_muestra]
    cond_contor = {
        'A': registro['valor_A'],
        'B': registro['valor_B'],
        'C': registro['valor_C'],
        'D': registro['valor_D']
    }
    typ_cond_contorno = {
        'A': registro['tipo_A'],
        'B': registro['tipo_B'],
        'C': registro['tipo_C'],
        'D': registro['tipo_D']
    }
    hot_point = {
        'T': registro['T_hp'],
        'i': int(registro['i_hp']),
        'j': int(registro['j_hp'])
    }
    material_nombre = registro['material']
    if mostrar:
        mostrar_tabla_variables_ordenada(cond_contor, typ_cond_contorno, hot_point, material_nombre)

    comparar_MT(y_results[0], y_results[1], Y_true_img,folder=f"{folder_resultsM[0]} vs {folder_resultsM[1]}",etiquetas=(f'{folder_resultsM[0]}', f'{folder_resultsM[1]}'),auto=auto)


