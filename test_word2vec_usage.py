from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from Utils.WordNormalizer import WordNormalizer

vector_path = 'outputs/w2v/w2v_model_minCount10_vectors.csv'

vocab_vectors = np.loadtxt(vector_path, delimiter=' ', dtype='str', comments=None, skiprows=1)

vocab = vocab_vectors[:, 0]
vectors = vocab_vectors[:, 1:].astype('float')

# print(vocab)
# print(vectors)
nor = WordNormalizer()
nor.read_resource()

tags = set()
with open('inputs/searched_tags.tsv') as f:
    lines = f.read().split('\n')
    for line in lines:
        tag = line.split('	')[0]
        normalized_tag = '#' + nor.normalize(tag)
        tags.add(normalized_tag)

# print(tags)

index_of_tags = np.where(np.isin(vocab, list(tags)))
# print(index_of_tags)

searched_vocab = vocab[index_of_tags]
searched_vectors = vectors[index_of_tags]

dist = euclidean_distances(searched_vectors, searched_vectors)
# print(dist)

# edges_index = np.where((dist < 1.2) & (dist !=0))
#
# print(len(edges_index[0]))
#
# # print(edges_index)
# source_nodes = searched_vocab[edges_index[0]]
# target_nodes = searched_vocab[edges_index[1]]
#
# edges = np.array([source_nodes,target_nodes]).transpose()
#
# print(edges.shape)
# print(edges)

kmeans = KMeans(n_clusters=100, random_state=0).fit(searched_vectors)
print(kmeans.labels_)

x = np.array([kmeans.labels_, searched_vocab]).transpose()
with np.printoptions(threshold=np.inf):
    print(x[x[:, 0].argsort()])
