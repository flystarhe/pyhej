'''https://github.com/fchollet/keras/blob/master/examples/cifar10_resnet.py

ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
'''
import os
import numpy as np
from easydict import EasyDict as edict


cfg = edict()
# Training parameters
cfg.data = '/your/image/path'
cfg.classes = 10
cfg.batch_size = 32
cfg.epochs = 200
# Model parameter
cfg.n = 3
cfg.version = 2  # 1(ResNet v1), 2(ResNet v2)
cfg.model_type = 'ResNet%dv%d' % (cfg.n * 6 + 2, cfg.version)
# Other
cfg.save_dir = '/tmps/cifar10'


if not os.path.isdir(cfg.save_dir):
    os.makedirs(cfg.save_dir)
if os.path.isdir(cfg.data):
    from pyhej.keras.classifier.tools import loader_from_directory
    train_data = loader_from_directory(os.path.join(cfg.data, 'train'), target_size=(32, 32), batch_size=32)
    steps_per_epoch = int(np.ceil(train_data.n/train_data.batch_size))
    validation_data = loader_from_directory(os.path.join(cfg.data, 'val'), target_size=(32, 32), batch_size=32)
    validation_steps = int(np.ceil(validation_data.n/validation_data.batch_size))
else:
    # Load the CIFAR10 data.
    import keras
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, cfg.classes)
    y_test = keras.utils.to_categorical(y_test, cfg.classes)
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_data = datagen.flow(x_train, y_train, batch_size=batch_size)
    steps_per_epoch = int(np.ceil(train_data.n/train_data.batch_size))
    validation_data = datagen.flow(x_test, y_test, batch_size=batch_size)
    validation_steps = int(np.ceil(validation_data.n/validation_data.batch_size))


def lr_schedule(epoch):
    '''Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    '''
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


# Create model
from pyhej.keras.classifier.models import get_model
model = get_model('resnet_v{}'.format(cfg.version), cfg)


from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])


# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=os.path.join(cfg.save_dir, '%s_model.{epoch:03d}.h5' % cfg.model_type),
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]
model.fit_generator(train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=cfg.epochs,
                    validation_data=validation_data,
                    validation_steps=validation_steps,
                    callbacks=callbacks)