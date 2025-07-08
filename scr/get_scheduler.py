import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, factor=0.5, patience=50, min_lr=1e-6, verbose=True):

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )
    
    original_step = scheduler.step

    def step(val_loss):
        old_lr = optimizer.param_groups[0]['lr']
        original_step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if verbose and new_lr < old_lr:
            print(f"🔻 Scheduler: LR reducido de {old_lr:.6e} a {new_lr:.6e} por falta de mejora en val_loss.")

    scheduler.step = step

    return scheduler
