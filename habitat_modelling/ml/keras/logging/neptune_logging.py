import neptune

from keras.callbacks import Callback

class NeptuneMonitor(Callback):
    def on_epoch_end(self, epoch, logs={}):
        neptune.send_metric('acc', epoch, logs['acc'])
        neptune.send_metric('loss', epoch, logs['loss'])
