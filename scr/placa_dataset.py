import numpy as np
from torch.utils.data import Dataset

class PlacaDataset(Dataset):
    
    def __init__(self, X_path, Y_path):

        self.X = np.load(X_path).astype(np.float32)
        self.Y = np.load(Y_path).astype(np.float32)

        self.mean_X = self.X.mean(axis=0)
        self.std_X = self.X.std(axis=0)

        self.std_X[self.std_X == 0] = 1.0  # Evitar división por cero
        
        self.X = (self.X - self.mean_X) / self.std_X

        self.mean_Y = self.Y.mean()
        self.std_Y = self.Y.std()

        #self.std_Y[self.std_Y == 0] = 1.0  # Evitar división por cero

        # Evitar división por cero
        if self.std_Y == 0:
            self.std_Y = 1.0

        self.Y = (self.Y - self.mean_Y) / self.std_Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]