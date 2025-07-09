from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from placa_dataset import PlacaDataset
from mlp_temp_regressor import MLPTempRegressor
from EarlyStopping import EarlyStopping

BASE_DIR = Path().resolve()

# def MSEcustom_batch0(outputs, Y_batch, X_batch, radio, alfa, mean_X, std_X):

#     '''
#         Esta funcion calcula el MSE de manera ponderada, se pone enfasis en la pocision del punto caliente 
#         que forma parte de los datos de entreda X del modelo. 

#         Los parametros - radio - y - alfa - definen que tanta importancia va a tener este punto caliente

#         Ingresa: 
#             outputs: Matriz de predicciones para el batch que se esta considerando - (batch_size,2500)
#                      Serian los valores que calculo el modelo
#             X_batch, Y_batch: Datos del batch
#             radio (int): Es la cantidad de celdas que me muevo desde el punto caliente en todas direcciones (Se construye un cuadrado)
#             alfa (float 0<alfa<1): Es el parametro que define el peso del punto caliente
#             mean_X, std_Y: Son los parametros para desnormalizar 

#     '''
#     #   ....... PARAMETROS QUE DEFINEN LA PLACA ...........cuando saco los valores de X-bach ... de i,j estos salen normalizados ... y representan las pociciones 


#     Nx = Ny = 50                    #   Cant. de nodos
#     #dx = dy = 0.05                  #   Espaciado entre nodos [m]

#     #   ...... PARAMETROS DE LOS OUTPUTS ..................

#     device = outputs.device         #   Dispocitivo que se esta usando - CPU en mi caso
#     batch_size, N = outputs.shape   #   Dimendiones de los outputs

#     n_radio = int(radio)       #   Para convertir el radio [m] a numero de celdas 

#     #   ...... ACONDICIONAMIENTO DE LOS PUNTOS .......................
#     #   Se toman todas la filas de la columna 2,3 (Que representan los valores i,J del punto caliente en cada muestra del batch), y se desnormaliza
    
#     i_hp_batch = (X_batch[:, 2] * std_X[2] + mean_X[2]).round().long()
#     j_hp_batch = (X_batch[:, 3] * std_X[3] + mean_X[3]).round().long()

#     # Construir máscara (batch_size, N) en True donde está dentro del radio
#     hot_mask = torch.zeros((batch_size, N), dtype=torch.bool, device=device)

#     # Se itera sobre todas las muestras del batch
#     for b in range(batch_size):
#         #   Tomo la pocision del punto caliente de cada muestra en el batch 
#         i_hp = i_hp_batch[b]
#         j_hp = j_hp_batch[b]

#         #   Genero mi "cuadrado/ventana" al rededor del punto y tomo los limites, se verifica no tomar puntos por fuera de la placa
#         i_min = max(0, i_hp - n_radio)
#         i_max = min(Nx - 1, i_hp + n_radio)
#         j_min = max(0, j_hp - n_radio)
#         j_max = min(Ny - 1, j_hp + n_radio)

#         #   Itero sobre cada posicion del "cuadrado" y guardo "true" en la mascara para cada pocision
#         for j in range(j_min, j_max + 1):
#             for i in range(i_min, i_max + 1):
#                 idx = j * Nx + i
#                 hot_mask[b, idx] = True

#     #   Calculo el error para los puntos dentro y fuera del cuadrado y les doy un peso con el coeficiente alfa
#     loss_radio = ((outputs - Y_batch) ** 2)[hot_mask].mean()
#     loss_resto = ((outputs - Y_batch) ** 2)[~hot_mask].mean()

#     loss = (1 - alfa) * loss_resto + alfa * loss_radio

#     return loss

def MSEcustom_batch(outputs, Y_batch, X_batch, radio, alfa, mean_X, std_X):

    Nx = Ny = 50

    device = outputs.device
    batch_size, N = outputs.shape
    n_radio = int(radio)

    i_hp_batch = (X_batch[:, 2] * std_X[2] + mean_X[2]).round().long()
    j_hp_batch = (X_batch[:, 3] * std_X[3] + mean_X[3]).round().long()

    offset = torch.arange(-n_radio, n_radio + 1, device=device)             #   Genero un vector de enteros. Con n_radio = 1 -> [-1, 0 , 1]
    offset_i, offset_j = torch.meshgrid(offset, offset, indexing='ij')      #   Genero todas la combinaciones 2D de forma vectorizada 

    offset_i = offset_i.flatten()
    offset_j = offset_j.flatten()

    #   Genero todos los indices alrededor del punto caliente 

    i_idx = i_hp_batch[:, None] + offset_i[None, :]
    j_idx = j_hp_batch[:, None] + offset_j[None, :]

    #   Elimino todos los indices que estan fuera del rango de la placa

    i_idx = i_idx.clamp(0, Nx - 1)
    j_idx = j_idx.clamp(0, Ny - 1)

    #   Obtengo a partir de las coordenadas i,j los indices lineales correspondientes

    idx_flat = (j_idx * Nx + i_idx).reshape(batch_size, -1)

    hot_mask = torch.zeros((batch_size, N), dtype=torch.bool, device=device)
    hot_mask.scatter_(1, idx_flat, True)

    loss_radio = ((outputs - Y_batch) ** 2)[hot_mask].mean()
    loss_resto = ((outputs - Y_batch) ** 2)[~hot_mask].mean()

    loss = (1 - alfa) * loss_resto + alfa * loss_radio
    #loss = loss_resto + alfa * loss_radio

    return loss, loss_radio, loss_resto
    

def entrenar_modelo(X_path, Y_path, X_val_path, Y_val_path, subfolder_name, epochs, batch_size, lr, radio, alfa):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset de entrenamiento ---------------------------------------------------------

    dataset = PlacaDataset(X_path, Y_path)

    mean_X = torch.tensor(dataset.mean_X, device=device)
    std_X = torch.tensor(dataset.std_X, device=device)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    #------------------------------------------------------------------------------------

    # Dataset de validación ------------------------------------------------------------
    
    val_dataset = PlacaDataset(X_val_path, Y_val_path)

    mean_X_val = torch.tensor(val_dataset.mean_X, device=device)
    std_X_val = torch.tensor(val_dataset.std_X, device=device)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #------------------------------------------------------------------------------------

    input_dim = dataset.X.shape[1]
    output_dim = dataset.Y.shape[1]

    model = MLPTempRegressor(input_dim, output_dim).to(device)
    
    #................................................................criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #------------------------------------------------------------------------------------

    early_stopper = EarlyStopping(patience=100, min_delta=5e-3, verbose=True)

    loss_history = []
    loss_radio_history = []
    loss_resto_history = []
    
    val_loss_history = []
    val_loss_radio_history = []
    val_loss_resto_history = []


    for epoch in range(epochs):
    
        model.train()
        
        running_loss = 0.0
        running_loss_radio = 0.0
        running_loss_resto = 0.0
        
        for X_batch, Y_batch in dataloader:
        
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            
            outputs = model(X_batch)

            #............................................loss = criterion(outputs, Y_batch)

            loss, loss_radio, loss_resto = MSEcustom_batch(outputs, Y_batch, X_batch, radio, alfa, mean_X, std_X)
            
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
            running_loss_radio += loss_radio.item()
            running_loss_resto += loss_resto.item()

        avg_loss = running_loss / len(dataloader)
        avg_loss_radio = running_loss_radio / len(dataloader)
        avg_loss_resto = running_loss_resto / len(dataloader)

        
        loss_history.append(avg_loss)
        loss_radio_history.append(avg_loss_radio)
        loss_resto_history.append(avg_loss_resto)

        
        model.eval()
        
        val_running_loss = 0.0
        val_running_loss_radio = 0.0
        val_running_loss_resto = 0.0
        
        with torch.no_grad():
        
            for X_batch_val, Y_batch_val in val_loader:
            
                X_batch_val = X_batch_val.to(device)
                Y_batch_val = Y_batch_val.to(device)

                outputs_val = model(X_batch_val)

                val_loss, val_loss_radio, val_loss_resto =  MSEcustom_batch(outputs_val, Y_batch_val, X_batch_val, radio, alfa, mean_X_val, std_X_val)
                
                val_running_loss += val_loss.item()
                val_running_loss_radio += val_loss_radio.item()
                val_running_loss_resto += val_loss_resto.item()

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_loss_radio = val_running_loss_radio / len(val_loader)
        avg_val_loss_resto = val_running_loss_resto / len(val_loader)
        
        val_loss_history.append(avg_val_loss)
        val_loss_radio_history.append(avg_val_loss_radio)
        val_loss_resto_history.append(avg_val_loss_resto)

        #..................................................................
        early_stopper(avg_val_loss)
        #..................................................................

        if early_stopper.should_stop:
            print(f"Entrenamiento detenido en epoch {epoch} por Early Stopping.")
            break

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Val_Loss: {avg_val_loss:.6f}")
            
    #------------------------------------------------------------------------------------------------------

    # Guardado de resultados y modelo
    base_folder = BASE_DIR.parent / 'results'
    save_folder = base_folder / subfolder_name if subfolder_name else base_folder

    if not save_folder.exists():
        save_folder.mkdir(parents=True)
        print(f'Se creó la carpeta: {save_folder}')

    np.save(save_folder / 'mean_X.npy', dataset.mean_X)
    np.save(save_folder / 'std_X.npy', dataset.std_X)
    np.save(save_folder / 'mean_Y.npy', dataset.mean_Y)
    np.save(save_folder / 'std_Y.npy', dataset.std_Y)

    np.save(save_folder / 'val_mean_X.npy', val_dataset.mean_X)
    np.save(save_folder / 'val_std_X.npy', val_dataset.std_X)
    np.save(save_folder / 'val_mean_Y.npy', val_dataset.mean_Y)
    np.save(save_folder / 'val_std_Y.npy', val_dataset.std_Y)

    torch.save(model.state_dict(), save_folder / 'model_train.pt')
    torch.save(model, save_folder / 'model_train_complete.pt')

    np.save(save_folder / 'loss_history_Train.npy', np.array(loss_history))
    np.save(save_folder / 'loss_radio_history_Train.npy', np.array(loss_radio_history))
    np.save(save_folder / 'loss_resto_history_Train.npy', np.array(loss_resto_history))

    np.save(save_folder / 'loss_history_Val.npy', np.array(val_loss_history))
    np.save(save_folder / 'loss_radio_history_Val.npy', np.array(val_loss_radio_history))
    np.save(save_folder / 'loss_resto_history_Val.npy', np.array(val_loss_resto_history))

    current_lr = optimizer.param_groups[0]['lr']    # Es para tomar el ultimo Ir modificado

    params = {

        'epochs': epochs,
        'batch_size': batch_size,
        'lr_initial': lr,
        'lr_final': current_lr,
        'input_dim': input_dim,
        'output_dim': output_dim
    }

    np.save(save_folder / 'training_params.npy', params)

    print(f"Entrenamiento completado. Modelo guardado en {save_folder / 'model_train.pt'}")
