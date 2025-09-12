from pathlib import Path
import torch

class EarlyStopping:
    def __init__(self, patience=5, mode='max', monitor='val_macro_f1', save_dir='./runs/exp'):
        assert mode in ['max','min']
        self.patience = patience
        self.mode = mode
        self.monitor = monitor
        self.best = None
        self.best_epoch = None
        self.num_bad = 0
        self.ckpt_path = Path(save_dir)/'best.pth'

    def step(self, value, model, epoch):
        improved = False
        if self.best is None:
            improved = True
        else:
            if self.mode == 'max':
                improved = value > self.best
            else:
                improved = value < self.best
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, self.ckpt_path)
            return False  # don't stop
        else:
            self.num_bad += 1
            return self.num_bad > self.patience

    def load_best(self, model, map_location=None):
        ckpt = torch.load(self.ckpt_path, map_location=map_location)
        model.load_state_dict(ckpt['state_dict'])
        return ckpt.get('epoch', None)