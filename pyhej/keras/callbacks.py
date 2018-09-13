import re
import numpy as np
from keras.callbacks import Callback


class SaveBest(Callback):
    def __init__(self, filepath, monitor='val_.*?acc', mode='max'):
        super(SaveBest, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        tmps = []
        for key, val in logs.items():
            if re.match(self.monitor, key):
                tmps.append(val)
        current = sum(tmps)/len(tmps)
        if self.monitor_op(current, self.best):
            filepath = self.filepath.format(epoch=epoch, best=current, **logs)
            self.model.save(filepath, overwrite=True)
            self.best = current