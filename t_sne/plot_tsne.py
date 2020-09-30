import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import kaldi_io


def tsne_plotter(data, label, save_png, title):
    n_labels = len(set(label))
    tsne = TSNE(n_components=2, init='pca', learning_rate=10, perplexity=12, n_iter=1000)
    transformed_data = tsne.fit_transform(data)

    plt.figure()
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], 10, c=label, cmap=plt.cm.Spectral, alpha=0.5)
    plt.title(title)
    plt.savefig(save_png)


def sample_data(data_dir):
    # TODO

    data = np.array(data)
    label = np.array(label)
    return data, label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="file of npz data.")
    parser.add_argument("--pic_saved_path", type=str, default="tsne.png", 
                        help="file of saved png.")
    parser.add_argument("--pic_title", type=str, default="t-sne", 
                        help="title of png.")
    parser.add_argument("--n_speaker", type=int, default=10)
    parser.add_argument("--n_frame", type=int, default=10)

    args = parser.parse_args()

    print("loading data...")
    data, label = sample_data(args.data_dir, args.n_speaker, args.n_frame)

    print("ploting...")
    tsne_plotter(data, label, args.save_png, args.title)

