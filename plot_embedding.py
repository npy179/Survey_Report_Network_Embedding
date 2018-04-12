#!/usr/bin/python
from __future__ import print_function
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_sdne(embedding_name, labels_name):

    embedding = np.load(embedding_name)

    labels_dict = {}
    with open(labels_name) as fin:
        for line in fin:
            line_split = line.strip().split("\t")
            labels_dict[int(line_split[0])] = int(line_split[1])


    embedding = TSNE(n_components=2).fit_transform(embedding)
    fig, ax = plt.subplots(figsize=(15,15))

    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_dict.values(), alpha=0.5)
    ax.tick_params(direction='in', length=4, width=2, labelsize=20)
    plt.tight_layout()

    plt.savefig("sdne_tsne.png")

def plot_deepwalk(embedding_name, labels_name):

    labels_dict = {}
    with open(labels_name) as fin:
        for line in fin:
            line_split = line.strip().split("\t")
            labels_dict[int(line_split[0])] = int(line_split[1])

    labels = []
    embedding_matrix_list = []

    with open(embedding_name) as fin:
        header = fin.readline()
        for line in fin:
            line_split = [float(i) for i in line.strip().split(" ")]
            embedding_matrix_list.append(line_split[1:])
            labels.append(labels_dict[line_split[0]])


    embedding = np.asarray(embedding_matrix_list)
    embedding = TSNE(n_components=2).fit_transform(embedding)
    fig, ax = plt.subplots(figsize=(15,15))

    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, alpha=0.5)
    ax.tick_params(direction='in', length=4, width=2, labelsize=20)
    plt.tight_layout()
    plt.savefig("deepwalk_tsne.png")



def plot_MMDW(embedding_name, labels_name):

    labels_dict = {}
    with open(labels_name) as fin:
        for line in fin:
            line_split = line.strip().split("\t")
            labels_dict[int(line_split[0])] = int(line_split[1])

    embedding_matrix_list = []
    with open(embedding_name) as fin:
        for line in fin:
            line_split = [float(i) for i in line.strip().split(" ")]
            embedding_matrix_list.append(line_split[1:])

    embedding = np.asarray(embedding_matrix_list)
    embedding = TSNE(n_components=2).fit_transform(embedding)
    fig, ax = plt.subplots(figsize=(15,15))

    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_dict.values(), alpha=0.5)
    ax.tick_params(direction='in', length=4, width=2, labelsize=20)

    plt.tight_layout()
    plt.savefig("mmdw_tsne.png")


def main():
    embedding_name = sys.argv[1]
    labels_name = sys.argv[2]
    #plot_MMDW(embedding_name, labels_name)
    #plot_deepwalk(embedding_name, labels_name)
    plot_sdne(embedding_name, labels_name)

if __name__=="__main__":
    main()
