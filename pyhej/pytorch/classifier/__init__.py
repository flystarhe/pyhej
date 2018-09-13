# input shape
#   [*] Xception: 224x224
#   [-] VGG: 224x224, VGG16, VGG19
#   [-] ResNet: 224x224, ResNet50
#   [*] InceptionV3: 299x299
#   [-] InceptionResNetV2: 299x299
#   [-] MobileNet: 224x224
#   [keras/applications](https://keras.io/applications/)
#   [keras-cn/applications](http://keras-cn.readthedocs.io/en/latest/other/application/)
#   [pytorch/models](https://github.com/pytorch/vision/tree/master/torchvision/models)


################################################################
# pytorch
#   inputs: nSamples x nChannels x Height x Width
################################################################
# pytorch.models
#   url: https://github.com/pytorch/vision/blob/master/docs/source/models.rst
#   url: https://github.com/pytorch/examples/blob/master/imagenet/main.py


'''
import torch
import torchvision.models as models
# model_name: resnet18, resnet34, resnet50, resnet152, inception_v3 ..
# pretrained: model.load_state_dict('https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth')
model = models.__dict__[model_name](pretrained=True, num_classes=1000)
if use_cuda:
    model = torch.nn.DataParallel(model).cuda()
'''


# pytorch.transforms
#   url: https://github.com/pytorch/vision/blob/master/docs/source/transforms.rst


# pytorch.datasets
#   url: https://github.com/pytorch/vision/blob/master/docs/source/datasets.rst


