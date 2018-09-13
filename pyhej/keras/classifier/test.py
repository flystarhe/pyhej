import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


input_shape = (224, 224)
model = ResNet50(weights='imagenet')


# case zero
imgs = ['1.jpg', '2.jpg', '3.jpg']

inputs = []
for i, img in enumerate(imgs):
    img = image.load_img(img, target_size=input_shape)
    x = image.img_to_array(img)
    inputs.append(x)

inputs = np.array(inputs)
inputs = preprocess_input(inputs)
preds = model.predict(inputs)
decode_predictions(preds, top=3)


# case one
imgs = ['1.jpg', '2.jpg', '3.jpg']

inputs = np.zeros((len(imgs),) + input_shape, dtype='float32')
for i, img in enumerate(imgs):
    img = image.load_img(img, target_size=input_shape)
    x = image.img_to_array(img)
    inputs[i] = x

inputs = preprocess_input(inputs)
preds = model.predict(inputs)
decode_predictions(preds, top=3)


# case two
img = '1.jpg'

img = image.load_img(img, target_size=input_shape)
x = image.img_to_array(img)
inputs = np.expand_dims(x, axis=0)

inputs = preprocess_input(inputs)
preds = model.predict(inputs)
decode_predictions(preds, top=3)