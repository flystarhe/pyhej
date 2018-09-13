# https://github.com/mdbloice/Augmentor
#   pip install Augmentor
import Augmentor
import torchvision


def samples_base(source_directory, output_directory, save_format="jpg"):
    # 实例化一个Pipeline对象
    p = Augmentor.Pipeline(source_directory, output_directory, save_format)
    # 将操作添加到管道中
    p.resize(probability=1, width=512, height=512)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.class_labels
    p.status()
    # 根据您的规格生成10幅增强图像
    p.sample(10)


def samples_parallel(source_directory, output_directory, ground_truth, save_format="jpg"):
    # 图像可以通过管道以两个或两个以上的组合的方式传递
    # 以便可以对`ground_truth`数据进行相同的增强
    p = Augmentor.Pipeline(source_directory, output_directory, save_format)
    # 指向包含`ground_truth`数据的目录
    # 具有相同文件名的图像将被添加为`ground_truth`数据
    p.ground_truth(ground_truth)
    # 将操作添加到管道中
    p.resize(probability=1, width=512, height=512)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.class_labels
    p.status()
    # 根据您的规格生成10幅增强图像
    p.sample(10)


# Elastic Distortions(弹性扭曲)
def samples_distortions(source_directory, output_directory, save_format="jpg"):
    # 实例化一个Pipeline对象
    p = Augmentor.Pipeline(source_directory, output_directory, save_format)
    # 将操作添加到管道中
    p.resize(probability=1, width=512, height=512)
    # between 2 and 10 for the grid width and height
    # magnitude of between 1 and 10
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.class_labels
    p.status()
    p.sample(10)


# Perspective Transforms(透视变换)
def samples_perspective(source_directory, output_directory, save_format="jpg"):
    pass


# Generators for Keras and PyTorch
def samples_keras(source_directory, output_directory, save_format="jpg"):
    # 实例化一个Pipeline对象
    p = Augmentor.Pipeline(source_directory, output_directory, save_format)
    # 将操作添加到管道中
    p.resize(probability=1, width=512, height=512)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.class_labels
    p.status()
    # for Keras
    g = p.keras_generator(batch_size=128)
    # import matplotlib.pyplot as plt
    # images, labels = next(g)
    # plt.imshow(images[0].reshape(28, 28), cmap="Greys");
    return g


def samples_pytorch(source_directory, output_directory, save_format="jpg"):
    # 实例化一个Pipeline对象
    p = Augmentor.Pipeline(source_directory, output_directory, save_format)
    # 将操作添加到管道中
    p.resize(probability=1, width=512, height=512)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.class_labels
    p.status()
    # for PyTorch
    transforms = torchvision.transforms.Compose(
        [
         p.torch_transform(),
         torchvision.transforms.ToTensor(),
        ]
    )
    return transforms