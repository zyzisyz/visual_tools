#!/bin/bash

python plot_tsne.py \
	--data_path data \
	--pic_title mfcc \
	--pic_saved_path tsne.png \
	--n_speaker 10 \
	--n_frame 10

