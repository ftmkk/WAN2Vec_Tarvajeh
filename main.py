import numpy as np
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from os import path
import sys, getopt

from Visualizer import Visualizer
from WAN import WAN


def read_word2vec(path):
    word_embeddings_labels = []
    min_col_count = 1000000
    with open(path) as f:
        for line in f.read().split('\n')[1:]:
            elements = line.split(' ')
            if len(elements) < 2:
                continue
            word_embeddings_labels.append(elements[0])
            min_col_count = min(min_col_count, len(elements))

    word_embeddings = np.loadtxt(path, usecols=tuple(range(1, min_col_count)), skiprows=1)
    return word_embeddings, word_embeddings_labels


def read_args(argv):
    outputfile = 'outputs/word2vec.csv'
    inputfile = 'inputs/allResponses3.csv'

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    return inputfile, outputfile


def main(argv):
    inputfile, outputfile = read_args(argv)

    if not path.exists(outputfile):
        wan = WAN(inputfile)
        print(wan.graph.number_of_edges())
        wan.prune_edge(min_weight=1)
        print(wan.graph.number_of_edges())
        wan.prune_node(min_freq=2)
        print(wan.graph.number_of_edges())
        wan.reverse_weight()

        node2vec = Node2Vec(wan.graph, dimensions=10, walk_length=20, num_walks=50, workers=4, temp_folder='temp',
                            p=0.01, q=0.01)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        model.wv.save_word2vec_format(outputfile)

        word_embeddings = model.wv.vectors
        word_embeddings_labels = model.wv.index2word
    else:
        word_embeddings, word_embeddings_labels = read_word2vec(outputfile)

    X_embedded = TSNE(n_components=2).fit_transform(word_embeddings)
    viz = Visualizer()
    viz.scatter_plot(X_embedded[:, 0], X_embedded[:, 1], word_embeddings_labels)

    # simulating a pandas df['type'] column

    # # Save model for later use
    # model.save('outputs/word2vec_model')
    #
    # keyword = 'پیشرفت'
    # # Look for most similar nodes
    # print(model.wv.most_similar(keyword))  # Output node names are always strings
    # s = nx.neighbors(pruned_wan, keyword)
    # for e in s:
    #     print(e, s[e])


if __name__ == "__main__":
    main(sys.argv[1:])
