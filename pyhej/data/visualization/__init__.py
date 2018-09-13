import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_polar(theta, radius, preds=None, title=None, cycle=True):
    theta = [i*np.pi/180 for i in theta]
    rmin = min(radius)*0.9
    rmax = max(radius)*1.1
    if cycle:
        theta.append(theta[0])
        radius.append(radius[0])
    ax = plt.subplot(111, projection="polar")
    ax.plot(theta, radius)
    if preds is not None:
        size = len(preds)
        colors = range(size)
        radius = [max(radius)*1.1] * size
        ax.scatter(preds, radius, c=colors, cmap="hsv", alpha=0.75)
    if title is not None:
        ax.set_title(title, va="bottom")
    ax.set_rmin(min(rmin, min(radius)*0.9))
    ax.set_rmax(max(rmax, max(radius)*1.1))
    ax.set_xticks([i*15/180*np.pi for i in range(24)])
    ax.grid(True)
    plt.show()


def plot_hist(x, kdeplot=False):
    if kdeplot:
        sns.kdeplot(x, shade=True);
    else:
        sns.distplot(x, hist=True);