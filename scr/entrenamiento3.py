import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from placa_dataset import PlacaDataset
from mlp_temp_regressor import MLPTempRegressor

BASE_DIR = Path().resolve()

def entrenar_modelo(X_path, Y_path, X_val_path, Y_val_path, subfolder_name, epochs, batch_size, lr):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset de entrenamiento ---------------------------------------------------------
    
    dataset = PlacaDataset(X_path, Y_path)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    #------------------------------------------------------------------------------------

    # Dataset de validación ------------------------------------------------------------
    
    val_dataset = PlacaDataset(X_val_path, Y_val_path)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #------------------------------------------------------------------------------------

    input_dim = dataset.X.shape[1]
    output_dim = dataset.Y.shape[1]

    model = MLPTempRegressor(input_dim, output_dim).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #------------------------------------------------------------------------------------
	
    loss_history = []
    val_loss_history = []


    for epoch in range(epochs):
    
        model.train()
        
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

        
        model.eval()
        
        val_running_loss = 0.0
        
        with torch.no_grad():
        
            for X_batch_val, Y_batch_val in val_loader:
            
                X_batch_val = X_batch_val.to(device)
                Y_batch_val = Y_batch_val.to(device)

                outputs_val = model(X_batch_val)
                val_loss = criterion(outputs_val, Y_batch_val)
                
                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        
        val_loss_history.append(avg_val_loss)

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
    np.save(save_folder / 'loss_history_Val.npy', np.array(val_loss_history))

    params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'input_dim': input_dim,
        'output_dim': output_dim
    }
    np.save(save_folder / 'training_params.npy', params)

    print(f"Entrenamiento completado. Modelo guardado en {save_folder / 'model_train.pt'}")

