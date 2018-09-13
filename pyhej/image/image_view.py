import matplotlib.pyplot as plt
from pyhej.image import load_img


def image_show_path(imgs, col=5, height=1.0, target_size=None):
    row = int(len(imgs)/col) + 1
    plt.figure(figsize=(18, int(row/col*18*height)))
    for i in range(row):
        for j in range(col):
            num = i*col + j
            if len(imgs) > num:
                img = load_img(imgs[num], target_size=target_size)
                ax = plt.subplot(row, col, num + 1)
                ax.imshow(img)
                ax.set_axis_off()
            else:
                break
    plt.show()


def image_show_imgs(imgs, col=5, height=1.0, mode='rgb'):
    row = int(len(imgs)/col) + 1
    plt.figure(figsize=(18, int(row/col*18*height)))
    for i in range(row):
        for j in range(col):
            num = i*col + j
            if len(imgs) > num:
                img = imgs[num][:,:,::-1] if mode=='bgr' else imgs[num]
                ax = plt.subplot(row, col, num + 1)
                ax.imshow(img)
                ax.set_axis_off()
            else:
                break
    plt.show()