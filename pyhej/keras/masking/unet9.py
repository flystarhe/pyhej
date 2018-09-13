from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, Concatenate


def get_unet(input_shape, num_filters=32):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(filters=num_filters * 1, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = Conv2D(filters=num_filters * 1, kernel_size=3, padding='same', activation='relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=num_filters * 2, kernel_size=3, padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=num_filters * 2, kernel_size=3, padding='same', activation='relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=num_filters * 4, kernel_size=3, padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=num_filters * 4, kernel_size=3, padding='same', activation='relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filters=num_filters * 8, kernel_size=3, padding='same', activation='relu')(pool3)
    conv4 = Conv2D(filters=num_filters * 8, kernel_size=3, padding='same', activation='relu')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(filters=num_filters * 16, kernel_size=3, padding='same', activation='relu')(pool4)
    conv5 = Conv2D(filters=num_filters * 16, kernel_size=3, padding='same', activation='relu')(conv5)

    up1 = Concatenate(axis=-1)([conv4, UpSampling2D(size=(2, 2))(conv5)])
    conv6 = Conv2D(filters=num_filters * 8, kernel_size=3, padding='same', activation='relu')(up1)
    conv6 = Conv2D(filters=num_filters * 8, kernel_size=3, padding='same', activation='relu')(conv6)

    up2 = Concatenate(axis=-1)([conv3, UpSampling2D(size=(2, 2))(conv6)])
    conv7 = Conv2D(filters=num_filters * 4, kernel_size=3, padding='same', activation='relu')(up2)
    conv7 = Conv2D(filters=num_filters * 4, kernel_size=3, padding='same', activation='relu')(conv7)

    up3 = Concatenate(axis=-1)([conv2, UpSampling2D(size=(2, 2))(conv7)])
    conv8 = Conv2D(filters=num_filters * 2, kernel_size=3, padding='same', activation='relu')(up3)
    conv8 = Conv2D(filters=num_filters * 2, kernel_size=3, padding='same', activation='relu')(conv8)

    up4 = Concatenate(axis=-1)([conv1, UpSampling2D(size=(2, 2))(conv8)])
    conv9 = Conv2D(filters=num_filters * 1, kernel_size=3, padding='same', activation='relu')(up4)
    conv9 = Conv2D(filters=num_filters * 1, kernel_size=3, padding='same', activation='relu')(conv9)

    outputs = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=outputs)


'''
import os
import shutil
tmps_dir = 'tmps'
if os.path.isdir(tmps_dir):
    shutil.rmtree(tmps_dir)
os.system('mkdir -m 777 -p {}'.format(tmps_dir))


input_shape = (32, 32, 1)
num_filters = 8
model_path = os.path.join(tmps_dir, 'unet.hdf5')


from keras.callbacks import ModelCheckpoint
from pyhej.keras.masking.unet9 import get_unet
model = get_unet(input_shape, num_filters)
model.compile(optimizer='adam', loss='binary_crossentropy', matrics=['accuracy'])


imgs_train, mask_train = load_data(..., input_shape)
checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=0, save_best_only=True)
model.fit(imgs_train, mask_train, batch_size=8, epochs=3, callbacks=[checkpoint], validation_split=0.2)


import cv2 as cv
import numpy as np
from keras.models import load_model
from pyhej.keras.masking.utils import image_mask
model = load_model(filepath=model_path)
img = cv.imread('test.jpg', 0)
x = cv.resize(img, (input_shape[1], input_shape[0]), interpolation=cv.INTER_CUBIC)
x = np.expand_dims(x, -1)
y = model.predict(np.asarray([x]))
pred = y[0, ..., 0]


mask = cv.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)
masked = image_mask(img, mask>0.5)
output = np.concatenate([np.dstack((img, img, img)), masked * 255], 1)
cv.imwrite(os.path.join(save_dir, 'test_masked.jpg'), output)


## ksize = input_shape[0]
## img = cv.imread(img, 0)
## h, w = img.shape
## new_h = (h + ksize - 1) // ksize * ksize
## new_w = (w + ksize - 1) // ksize * ksize
## new_img = np.zeros((new_h, new_w, 1))
## new_img[:h, :w, 0] = img
## out_img = np.zeros(new_img.shape)
## for i in range(0, new_img.shape[0], ksize):
##     for j in range(0, new_img.shape[1], ksize):
##         x = new_img[i:i+ksize, j:j+ksize, :]
##         y = model.predict(np.asarray([x]))[0]
##         out_img[i:i+ksize, j:j+ksize, :] = y
## pred = out_img[:h, :w, 0]
'''