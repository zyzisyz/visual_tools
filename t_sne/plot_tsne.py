import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import kaldi_io
import scipy.io.wavfile


def get_rand_list(num, _range):
    rand_list = []
    while num != 0:
        x = np.random.randint(0, _range)
        if x not in rand_list:
            rand_list.append(x)
            num -= 1
    return rand_list

def tsne_plotter(data, label, save_png, title):
    n_labels = len(set(label))
    tsne = TSNE(n_components=2, init='pca', learning_rate=10, perplexity=12, n_iter=1000)
    transformed_data = tsne.fit_transform(data)

    plt.figure()
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], 10, c=label, cmap=plt.cm.Spectral, alpha=0.5)
    plt.title(title)
    plt.savefig(save_png)


def sample_data(data_dir, n_speaker, n_frame):
    # TODO
    f_speaker = os.listdir(data_dir)
    rand_list = get_rand_list(n_speaker, len(f_speaker))
    chosen_speakers = [f_speaker[idx] for idx in rand_list]
    wave_files = []
    for speaker in chosen_speakers:
        now_path = os.path.join(data_dir, speaker)
        total_frame = [] 
        for (root, dirs, files) in os.walk(now_path):
            total_frame += [root + f for f in files]
        rand_list = get_rand_list(n_frame, len(total_frame))
        frame_list = [total_frame[i] for i in rand_list]
        wave_files += frame_list

    data = []
    for name in wave_files:
        wav = scipy.io.wavfile.read(name) #read the waveform
        #TODO
        frames = mfcc(wav) #turn waveform to spectrogram

        frame = frames[np.randint(len(frames))] #select a frame in spectrogram
        data.append(frame)

    data = np.array(data)
    #label = np.array(label)
    label = np.repeat(range(n_speaker) , n_frame)
    return data, label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="file of npz data.")
    parser.add_argument("--pic_saved_path", type=str, default="tsne.png", 
                        help="file of saved png.")
    parser.add_argument("--pic_title", type=str, default="t-sne", 
                        help="title of png.")
    parser.add_argument("--n_speaker", type=int, default=10)
    parser.add_argument("--n_frame", type=int, default=10)

    args = parser.parse_args()

    print("loading data...")
    data, label = sample_data(args.data_path, args.n_speaker, args.n_frame)

    print("ploting...")
    tsne_plotter(data, label, args.save_png, args.title)

