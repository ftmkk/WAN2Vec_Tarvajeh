import multiprocessing
from time import time

from gensim.models import Word2Vec
from os import path

model_path = 'outputs/w2v/w2v_model_minCount10'

if not path.exists(model_path):
    w2v_model = Word2Vec(min_count=10,
                         window=4,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=10,
                         workers=multiprocessing.cpu_count() - 1)

    with open('inputs/tweets_tokens.txt') as f:
        sentences = [s.split(' ') for s in f.read().split('\n')]

    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=60, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.init_sims(replace=True)

    w2v_model.save(model_path)
    # w2v_model.save_word2vec_format('w2v_model_vectors.csv')
    w2v_model.wv.save_word2vec_format(model_path + '_vectors.csv')

w2v_model = Word2Vec.load(model_path)

print(w2v_model.wv.most_similar(positive=['مدرسه']))

# print(w2v_model.wv.similarity("فوتبال", 'توپ'))
