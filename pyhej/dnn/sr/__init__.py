import os
import h5py
import codecs
import shutil
import math
import numpy as np
from PIL import Image
import torch
import torch.autograd as autograd
import torch.utils.data as data
from torchvision.transforms import ToTensor
from pyhej.image.pillow import image_new, draw_text


def PSNR(pred, gt):
    imdff = pred - gt
    mse = np.mean(imdff**2)
    if mse == 0:
        return 100
    return 10 * math.log10(255.**2 / mse)


def colorize(y, ycbcr):
    '''
    ycbcr:
      from scipy.ndimage import imread
      ycbcr = imread(image_file, mode='YCbCr')
      y = ycbcr[:,:,0].astype(np.float32)
      cb = ycbcr[:,:,1].astype(np.float32)
      cr = ycbcr[:,:,2].astype(np.float32)
    '''
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, 'YCbCr').convert('RGB')
    return img


def load_img(img_l, img_h, upscale_factor=None):
    img_l = Image.open(img_l).convert('YCbCr')
    img_h = Image.open(img_h).convert('YCbCr')

    if upscale_factor:
        wid, hei = img_h.size
        if wid % upscale_factor or hei % upscale_factor:
            wid -= wid % upscale_factor
            hei -= hei % upscale_factor
            img_h = img_h.resize((wid, hei), Image.BICUBIC)

        wid, hei = wid//upscale_factor, hei//upscale_factor
        if img_l.size != (wid, hei):
            img_l = img_l.resize((wid, hei), Image.BICUBIC)

    img_l_y, _, _ = img_l.split()
    img_h_y, _, _ = img_h.split()

    return img_l_y, img_h_y


class DatasetFromH5(data.Dataset):
    def __init__(self, file_path, zero_center=False):
        super(DatasetFromH5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.label = hf.get('label')
        self.zero_center = zero_center

    def __getitem__(self, index):
        input = self.data[index,:,:,:]
        target = self.label[index,:,:,:]

        if self.zero_center:
            input = input - input.mean()
            target = target - target.mean()

        return torch.from_numpy(input).float(), torch.from_numpy(target).float()

    def __len__(self):
        return self.data.shape[0]


class DatasetFromFile(data.Dataset):
    def __init__(self, images, input_transform=None, target_transform=None, upscale_factor=None, zero_center=False):
        '''
        images:
          if str, a file path
          if list, such as [('img001_l.jpg','img001_h.jpg'), ..]
        '''
        if isinstance(images, str):
            self.images = []
            with codecs.open(images, 'r', 'utf-8') as reader:
                for i, line in enumerate(reader.readlines(), 1):
                    img_b, img_h = line.strip().split(',')
                    self.images.append((img_b, img_h))
        else:
            self.images = images

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.upscale_factor = upscale_factor
        self.zero_center = zero_center

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index
        Returns:
            tuple: (img_l, img_h)
        '''
        img_l, img_h = self.images[index]
        input, target = load_img(img_l, img_h, self.upscale_factor)

        if self.input_transform:
            input = self.input_transform(input)
        else:
            input = ToTensor()(input)

        if self.target_transform:
            target = self.target_transform(target)
        else:
            target = ToTensor()(target)

        if self.zero_center:
            input = input - input.mean()
            target = target - target.mean()

        return input, target

    def __len__(self):
        return len(self.images)


def adjust_lr_base(epoch, step=30, init_lr=0.01):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    '''
    lr = init_lr * (0.1 ** (epoch // step))
    return lr


def adjust_lr_opti(optimizer, epoch, step=30, init_lr=0.01, epsilon=1e-7):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    '''
    lr = init_lr * (0.1 ** (epoch // step))

    if lr < epsilon:
        lr = init_lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(cuda, epoch, model, criterion, optimizer, data_loader):
    model.train()
    avg_loss = 0

    for iteration, batch in enumerate(data_loader, 1):
        inputs, targets = autograd.Variable(batch[0]), autograd.Variable(batch[1])
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        #print('===> Epoch[{}]({}/{}): Loss: {:.6f}'.format(epoch, iteration, len(data_loader), loss.data[0]))
        avg_loss += loss.data[0]

    print('Epoch[{}]:\n===> Train: Avg Loss: {:.6f}'.format(epoch, avg_loss / len(data_loader)))
    return avg_loss / len(data_loader)


def test(cuda, epoch, model, criterion, data_loader):
    model.eval()
    avg_mse = 0

    for batch in data_loader:
        inputs, targets = autograd.Variable(batch[0]), autograd.Variable(batch[1])
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # if and only if `criterion = nn.MSELoss()`
        # `ToTensor()(PIL Image)`:
        #   Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
        #   to (C x H x W) in the range [0.0, 1.0]
        #psnr = 10 * math.log10(1 / loss.data[0])
        avg_mse += loss.data[0]

    print('===> Test : Avg MSE: {:.6f}'.format(avg_mse / len(data_loader)))
    return avg_mse / len(data_loader)


def save_checkpoint(state, save_dir, is_best=False, checkpoint='checkpoint.pth.tar'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fname = os.path.join(save_dir, checkpoint)
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, os.path.join(save_dir, 'model_best.pth.tar'))


def prediction(model, img_b, target_size=None, img_gt=None, cuda=False, zero_center=False):
    '''
    target_size:
      if None, the same as sub_pixel model
      if tuple of ints (width, height), the same as bicubic+dnn model
    '''
    img_b = Image.open(img_b).convert('YCbCr')
    y, cb, cr = img_b.split()

    if target_size:
        if target_size != y.size:
            y = y.resize(target_size, Image.BICUBIC)

    input = autograd.Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    input_center = input.mean()

    if zero_center:
        input = input - input_center

    if cuda:
        input = input.cuda()

    output = model(input)
    if cuda:
        output = output.cpu()

    if zero_center:
        output = output + input_center

    output = output.data[0].numpy()
    output *= 255.0
    output = output.clip(0, 255)

    out_img_y = Image.fromarray(np.uint8(output[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    if img_gt:
        img_gt = Image.open(img_gt).convert('YCbCr')
        y, cb, cr = img_gt.split()
        psnr = PSNR(output[0], np.asarray(y, dtype=np.float32))
    else:
        psnr = None

    return out_img, psnr


def imprint(img_h, img_b, img_gt=None, text=None, filename=None):
    '''
    img_h: A PIL Image
    img_b: A file path
    img_gt: A file path
    '''
    img_b = Image.open(img_b).convert('RGB')

    if img_gt:
        img_gt = Image.open(img_gt).convert('RGB')

        img_b = img_b.resize(img_gt.size, Image.BICUBIC)
        img_h = img_h.resize(img_gt.size, Image.BICUBIC)

        wid, hei = img_gt.size
        out = image_new((wid*3, hei), (0, 0, 0))
        out.paste(img_gt, (0, 0))
        out.paste(img_b , (0+wid, 0))
        out.paste(img_h , (0+wid+wid, 0))

        img_gt_y, _, _ = img_gt.convert('YCbCr').split()
        img_b_y, _, _ = img_b.convert('YCbCr').split()
        img_h_y, _, _ = img_h.convert('YCbCr').split()
        psnr_b = PSNR(np.asarray(img_gt_y, dtype=np.float32), np.asarray(img_b_y, dtype=np.float32))
        psnr_h = PSNR(np.asarray(img_gt_y, dtype=np.float32), np.asarray(img_h_y, dtype=np.float32))
    else:
        img_b = img_b.resize(img_h.size, Image.BICUBIC)

        wid, hei = img_h.size
        out = image_new((wid*2, hei), (0, 0, 0))
        out.paste(img_b, (0, 0))
        out.paste(img_h, (0+wid, 0))

        psnr_b = None
        psnr_h = None

    if text:
        draw_text(out, (5, 5), text, fill=(255, 0, 0))

    if filename:
        out.save(filename)
        return filename, psnr_b, psnr_h
    else:
        return out, psnr_b, psnr_h