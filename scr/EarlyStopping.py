class EarlyStopping:
    def __init__(self, patience=200, min_delta=1e-4, verbose=True):
        """
        patience: cuántas epochs esperar sin mejora antes de detener
        min_delta: mejora mínima considerada como mejora válida
        verbose: imprime mensajes cuando detiene
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            # ✅ Mostrar mensaje cada 20 pasos sin mejora
            if self.verbose and (self.counter % 20 == 0):
                print(f"EarlyStopping: {self.counter}/{self.patience} sin mejora (best_loss={self.best_loss:.6f})")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"🛑 Early stopping activado. Se detiene entrenamiento con best_loss={self.best_loss:.6f}")
                self.should_stop = True

