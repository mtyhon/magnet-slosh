#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.utils import linear_assignment_
from scipy.stats import itemfreq
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from itertools import chain
from sklearn.decomposition import PCA

import pdb

# Visualization
def moving_average(a, n=3) :
    # Adapted from http://stackoverflow.com/questions/14313510/does-numpy-have-a-function-for-calculating-moving-average
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_smooth(history, name):
    #pdb.set_trace()
    plt.plot(history, 'c', moving_average(history, 20), 'b')
    plt.savefig("results" + str(name) + '.svg')

def show_images(H):
    # make a square grid
    num = H.shape[0]
    rows = int(np.ceil(np.sqrt(float(num))))

    fig = plt.figure(1, [10, 10])
    grid = ImageGrid(fig, 111, nrows_ncols=[rows, rows])

    for i in range(num):
        grid[i].axis('off')
        grid[i].imshow(H[i], cmap='Greys')

    # Turn any unused axes off
    for j in range(i, len(grid)):
        grid[j].axis('off')


def plot_embedding(X, y, imgs=None, title=None, name=None, save_embed=False, filename=None, batch_builder=None):
    X_tsne_30 = TSNE(n_components=2, random_state=1337, perplexity=30).fit_transform(X)
    X_tsne_10 = TSNE(n_components=2, random_state=1337, perplexity=10).fit_transform(X)
    X_tsne_50 = TSNE(n_components=2, random_state=1337, perplexity=50).fit_transform(X)

    X_pca = PCA(n_components=2).fit_transform(X)

    # Adapted from http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    x_min_tsne_10, x_max_tsne_10 = np.min(X_tsne_10,0), np.max(X_tsne_10,0)
    X_tsne_10 = (X_tsne_10 - x_min_tsne_10)/(x_max_tsne_10-x_min_tsne_10)

    x_min_tsne_30, x_max_tsne_30 = np.min(X_tsne_30,0), np.max(X_tsne_30,0)
    X_tsne_30 = (X_tsne_30 - x_min_tsne_30)/(x_max_tsne_30-x_min_tsne_30)

    x_min_tsne_50, x_max_tsne_50 = np.min(X_tsne_50,0), np.max(X_tsne_50,0)
    X_tsne_50 = (X_tsne_50 - x_min_tsne_50)/(x_max_tsne_50-x_min_tsne_50)

    x_min_pca, x_max_pca = np.min(X_pca,0), np.max(X_pca,0)
    X_pca = (X_pca - x_min_pca)/(x_max_pca-x_min_pca)

    # Plot colors numbers
    plt.figure(figsize=(30,10))
    ax = plt.subplot(141)
    ax1 = plt.subplot(142)
    ax2 = plt.subplot(143)
    ax3 = plt.subplot(144)
    for i in range(X.shape[0]):
        # plot colored number
        # ax.text(X[i, 0], X[i, 1], str(y[i]),
        #          color=plt.cm.Set1(y[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
        ax.text(X_pca[i, 0], X_pca[i, 1], str(y[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
        ax1.text(X_tsne_10[i, 0], X_tsne_10[i, 1], str(y[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
        ax2.text(X_tsne_30[i, 0], X_tsne_30[i, 1], str(y[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
        ax3.text(X_tsne_50[i, 0], X_tsne_50[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    ax.set_title('PCA w/ 2 components')
    ax1.set_title('t-SNE Perplexity=10')
    ax2.set_title('t-SNE Perplexity=30')
    ax3.set_title('t-SNE Perplexity=50')
    # Add image overlays
    if imgs is not None and hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(imgs[i], cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.savefig("results/" + str(name) + '_pca_tsne.png')

    if save_embed:
        if batch_builder is not None:
            np.savez_compressed("embed/" + str(name) + '_tsne_embed', embed=X, label=y, filename=filename, centroids = batch_builder.centroids,
                                assignments = batch_builder.assignments)
        else:
            np.savez_compressed("embed/" + str(name) + '_tsne_embed', embed=X, label=y, filename=filename)


def zip_chain(a, b):
    return list(chain(*zip(a, b)))


def plot_metric(*args, **kwargs):

    name = args[0]
    plot_data = []
    for i in range(1, len(args), 2):
        metrics = args[i]
        d = [m[name] for m in metrics]
        color = args[i + 1]
        plot_data.extend(zip_chain(d, color * len(d)))

    plt.plot(*plot_data)
    if kwargs['title']:
        plt.title(kwargs['title'])
    plt.show()



# Evaluation

def compute_rand_index(emb, labels):
    """
    https://en.wikipedia.org/wiki/Rand_index
    """
    n = len(emb)
    k = np.unique(labels).size

    m = KMeans(k)
    m.fit(emb)
    emb_labels = m.predict(emb)

    agreements = 0
    for i, j in zip(*np.triu_indices(n, 1)):
        emb_same = emb_labels[i] == emb_labels[j]
        gt_same = labels[i] == labels[j]

        if emb_same == gt_same:
            agreements += 1

    return float(agreements) / (n * (n-1) / 2)


def unsupervised_clustering_accuracy(emb, labels):
    k = np.unique(labels).size
    kmeans = KMeans(n_clusters=k, max_iter=35, n_init=15, n_jobs=-1).fit(emb)
    emb_labels = kmeans.labels_
    G = np.zeros((k,k))
    for i in range(k):
        lbl = labels[emb_labels == i]
        uc = itemfreq(lbl)
        for uu, cc in uc:
            G[i,uu] = -cc
    A = linear_assignment_.linear_assignment(G)
    acc = 0.0
    for (cluster, best) in A:
        acc -= G[cluster,best]
    return acc / float(len(labels))
