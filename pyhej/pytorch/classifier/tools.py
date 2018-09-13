'''https://github.com/pytorch/examples/tree/master/imagenet
'''
import os
import shutil
import codecs
import torch
import numpy as np
from PIL import Image
from sklearn.preprocessing import label_binarize


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    '''Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    '''
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        images = []
        for i, target in enumerate(sorted(os.listdir(root))):
            for dirpath, _, filenames in sorted(os.walk(os.path.join(root, target))):
                for filename in filenames:
                    if is_image_file(filename):
                        images.append((os.path.join(dirpath, filename), i))

        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        '''
        filepath, target = self.images[index]
        with open(filepath, 'rb') as f:
            with Image.open(f) as img:
                input = img.convert('RGB')
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.images)


class DatasetFromFile(torch.utils.data.Dataset):
    def __init__(self, filename, transform=None, target_transform=None):
        images = []
        with codecs.open(filename, 'r', 'utf-8') as reader:
            for line in reader.readlines():
                if line.startswith('#'):
                    continue
                filepath, target = line.strip().split(',')
                images.append((filepath, target))

        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        '''
        filepath, target = self.images[index]
        with open(filepath, 'rb') as f:
            with Image.open(f) as img:
                input = img.convert('RGB')
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.images)


def get_mean_and_std(dataset, num_workers=4):
    '''Compute the mean and std value of dataset.
    import torchvision.transforms as transforms
    from pyhej.pytorch.classifier.tools import DatasetFromFolder, get_mean_and_std
    dataset = DatasetFromFolder('/your/image/path/', transforms.ToTensor())
    get_mean_and_std(dataset, 8)
    '''
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def test(model, inputs, maxk=1, softmax=None):
    '''test possible k categories
    import torch.nn as nn
    softmax = nn.Softmax()
    '''
    inputs_var = torch.autograd.Variable(inputs, volatile=True)
    outputs = model(inputs_var)
    if softmax:
        outputs = softmax(outputs)
    return outputs.topk(maxk, 1)


def eval(dataset, classes, model, batch_size=32, num_workers=8, softmax=True):
    '''evaluate model
    dataset = DatasetFromFolder('/your/image/path/')

    # if you need sequence of integer labels
    y_targets.argmax(1), y_outputs.argmax(1)

    # or you want to test topk
    y_targets[:, :k].sum(axis=1), y_outputs[:, :k].sum(axis=1)
    '''
    model.eval()
    if softmax:
        softmax = torch.nn.Softmax()
    y_targets, y_outputs = None, None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    for inputs, targets in dataloader:
        inputs_var = torch.autograd.Variable(inputs, volatile=True)
        outputs_var = model(inputs_var)
        if softmax:
            outputs_var = softmax(outputs_var)
        outputs = outputs_var.data.cpu()
        b_targets = label_binarize(targets.numpy(), classes)
        b_outputs = outputs.numpy()
        if y_targets is None:
            y_targets = b_targets
            y_outputs = b_outputs
        else:
            y_targets = np.r_[y_targets, b_targets]
            y_outputs = np.r_[y_outputs, b_outputs]
    return y_targets, y_outputs


class AverageMeter(object):
    '''Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, init_lr):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    '''
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(outputs, targets, topks=(1, 5)):
    '''Computes the precision@k for the specified values of k
    '''
    maxk = max(topks)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred)).float()

    accs = []
    for k in topks:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        accs.append(correct_k.mul_(100.0 / batch_size))
    return accs


def train(train_loader, model, criterion, optimizer, topks=(1, 5), use_cuda=True):
    train_loss = AverageMeter()
    train_accs = [AverageMeter() for _ in topks]

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(train_loader, 1):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        accs = accuracy(outputs.data, targets.data, topks)
        train_loss.update(loss.data[0], inputs.size(0))
        for k, acc in enumerate(accs):
            train_accs[k].update(acc[0], inputs.size(0))

    train_accs = [(k, acc.avg) for k, acc in zip(topks, train_accs)]
    text = ', '.join(['Acc@{}: {:.4f}'.format(k, acc) for k, acc in train_accs])
    print('  Train => Loss: {:.3f}, {}'.format(train_loss.avg, text))
    return train_accs


def validate(val_loader, model, criterion, topks=(1, 5), use_cuda=True):
    val_loss = AverageMeter()
    val_accs = [AverageMeter() for _ in topks]

    # switch to evaluate mode
    model.eval()

    for i, (inputs, targets) in enumerate(val_loader, 1):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs, volatile=True)
        targets = torch.autograd.Variable(targets, volatile=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        accs = accuracy(outputs.data, targets.data, topks)
        val_loss.update(loss.data[0], inputs.size(0))
        for k, acc in enumerate(accs):
            val_accs[k].update(acc[0], inputs.size(0))

    val_accs = [(k, acc.avg) for k, acc in zip(topks, val_accs)]
    text = ', '.join(['Acc@{}: {:.4f}'.format(k, acc) for k, acc in val_accs])
    print('  Valid => Loss: {:.3f}, {}'.format(val_loss.avg, text))
    return val_accs


def save_checkpoint(state, is_best, file_path='tmps'):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    file_name = os.path.join(file_path, 'checkpoint.pth.tar')
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(file_path, 'model_best.pth.tar'))