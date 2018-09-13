from keras.models import Model
from keras.layers import Dense, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from pyhej.keras.classifier.resnet import resnet_v1, resnet_v2


def get_model(model_name, cfg):
    '''about cfg:
    from easydict import EasyDict as edict
    cfg = edict()
    '''
    if model_name == 'inception_v3':
        # cfg.include_top = True/False
        # cfg.weights = 'imagenet'/None
        # cfg.classes = int, have classes
        model = InceptionV3(include_top=cfg.include_top, weights=cfg.weights, classes=cfg.classes)
        if not cfg.include_top:
            x = model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(cfg.classes, activation='softmax')(x)
            model = Model(inputs=model.input, outputs=x)
    elif model_name == 'xception':
        # cfg.include_top = True/False
        # cfg.weights = 'imagenet'/None
        # cfg.classes = int, have classes
        model = Xception(include_top=cfg.include_top, weights=cfg.weights, classes=cfg.classes)
        if not cfg.include_top:
            x = model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(cfg.classes, activation='softmax')(x)
            model = Model(inputs=model.input, outputs=x)
    elif model_name == 'mobilenet':
        # cfg.input_shape = None/(224, 224, 3)/..
        # cfg.alpha = 1.0
        # cfg.include_top = True/False
        # cfg.weights = 'imagenet'/None
        # cfg.classes = int, have classes
        model = MobileNet(cfg.input_shape, alpha=cfg.alpha, include_top=cfg.include_top, weights=cfg.weights, classes=cfg.classes)
    elif model_name == 'resnet_v1':
        # cfg.input_shape = (32, 32, 3)
        # cfg.n = 3  # number of sub-blocks, depth = n * 6 + 2
        # cfg.classes = int, have classes
        model = resnet_v1(cfg.input_shape, cfg.n*6+2, cfg.classes)
    elif model_name == 'resnet_v2':
        # cfg.input_shape = (32, 32, 3)
        # cfg.n = 3  # number of sub-blocks, depth = n * 6 + 2
        # cfg.classes = int, have classes
        model = resnet_v2(cfg.input_shape, cfg.n*6+2, cfg.classes)
    else:
        model = None
    return model