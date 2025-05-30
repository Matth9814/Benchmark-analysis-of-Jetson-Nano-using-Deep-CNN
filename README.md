## Benchmark-analysis-of-Jetson-Nano-using-Deep-CNN

### Introduction

This repository contains the analysis of a scientific paper prepared for the System-on-Chip Architecture course (M.Sc. in Computer Engineering, Embedded Systems – Polytechnic University of Turin).

### Usage

The repository is divided in two folders, for code and test results, and a [presentation](presentation.pdf), where the results are exposed and final remarks about the work are expressed. The [`results`](results) folder contains the results of various tests while [`code`](code) contains the code used to reproduce the experiment. The latter set of files is briefly described in the following: 

- `constants.py` is used to share variables between the files.

- `creat_datasets.py` contains some API functions that allow the creation of the custom dataset extracted from [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) in the correct format. Every function is provided with the necessary documentation.

- `data_processing.py` implements some of the input pipeline functions used in `CNN.py`. Each function is documented but for more information you can refer to the official Tensorflow documentation. 

- `CNN.py` needs Tensorflow in order to work. It contains the input pipeline for the custom dataset
and for datasets already provided by Tensorflow. The file is well commented so everything needed to make it work
should be written there. For information on Tensorflow functions, please refer to the official documentation.
Some tricky aspects about some functions are still commented for clarity.

- `k_means.py` is the implementation of the K-means algorithm in Tensorflow 2. It doesn't use the GPU on the
Jetson Nano so it is extremely slow. It takes the images from one of the custom datasets previously created as input.
It was used just to check how k-means distributes the images when they are divided in more than 13 classes.
These classes are not extracted from the file because the network was already showing low accuracy with other
fairly simple networks (e.g. fashion MNIST).

- `plot_res.py` plots training/validation accuracy and loss from the provided JSON files.
They are generated as output of `CNN.py` and their name has to be specified inside `plot_list` in order to parse them.

### License

Distributed under the MIT License. See `LICENSE` for more information.

### Disclaimer

This work is an **independent and personal analysis** of the original scientific article.  
The original paper and all its contents (text, figures, tables) are the **property of the respective authors and/or publisher**.

No copyrighted or proprietary content is redistributed here.

This analysis is provided **solely for educational and research purposes**.
