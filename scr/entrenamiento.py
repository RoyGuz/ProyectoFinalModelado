import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from placa_dataset import PlacaDataset
from mlp_temp_regressor import MLPTempRegressor

BASE_DIR = Path().resolve()

def entrenar_modelo(X_path,Y_path,subfolder_name,epochs,batch_size,lr):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PlacaDataset(X_path, Y_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]
    output_dim = dataset.Y.shape[1]

    model = MLPTempRegressor(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")


    #............................................................................................
    #............................................................................................
    # 
    #.................. CONFIGURACION PARA ALMACENAMIENTO ....................................... 

    # #   Defino la carpeta donde se guardaran los datos
    # base_folder = BASE_DIR.parent / 'results' 

    # if subfolder_name is not None:

    #     save_folder = save_folder = base_folder / subfolder_name

    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)
    #         print(f'Se creo la carpeta: {save_folder}')
    # else:
    #     save_folder = base_folder   #  Esto es por si no se indico ninguna ruta

    base_folder = BASE_DIR.parent / 'results'
    save_folder = base_folder / subfolder_name if subfolder_name else base_folder

    if not save_folder.exists():
        save_folder.mkdir(parents=True)
        print(f'Se creo la carpeta: {save_folder}')

    #   Guardado de los parametros de normalizacion

    np.save(save_folder / 'mean_X.npy', dataset.mean_X)
    np.save(save_folder / 'std_X.npy', dataset.std_X)
    np.save(save_folder / 'mean_Y.npy', dataset.mean_Y)
    np.save(save_folder / 'std_Y.npy', dataset.std_Y)

    #   Guardado del modelo entrenado

    torch.save(model.state_dict(), save_folder / 'model_train.pt')     # Esto solo guadaria los pesos del modelo .... 
    torch.save(model, save_folder / 'model_train_complete.pt')          # Esto guardaria el modelo completo .... 

    #   Guardado del historial de perdida (Con datos de entremaniento)

    np.save(save_folder / 'loss_history_Train.npy', np.array(loss_history))

    #   Guardado de los parametros de entrenamiento

    params = {
    'epochs': epochs,
    'batch_size': batch_size,
    'lr': lr,
    'input_dim': input_dim,
    'output_dim': output_dim
    }
    np.save(save_folder / 'training_params.npy', params)


    print(f"Entrenamiento completado el modelo quedo guardado en {save_folder / 'model_train.pt'}")
