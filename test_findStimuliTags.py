import numpy as np

from Utils.WordNormalizer import WordNormalizer


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


nor = WordNormalizer()
nor.read_resource()

with open('inputs/stimuliGroups_v2.csv') as f:
    stimuli = {
        '#' + nor.normalize(s.split(',')[1]).replace(' ', '_'): s.split(',')[0]
        for s in f.read().split('\n')}

with open('inputs/tweets_tokens.txt') as f:
    tweets = [s.split(' ') for s in f.read().split('\n')]

cntr = 0
subject_dist = {}
one_subject_dist = {}

selected_tweets = []
targets = []
for tweet in tweets:
    subject = set()
    tag_count = 0
    for token in tweet:
        if token in stimuli.keys():
            subject.add(stimuli[token])
        if token.startswith('#'):
            tag_count += 1
    subject_count = len(subject)
    if subject_count == 1 and tag_count <= 5:
        main_subject = subject.pop()
        if main_subject not in one_subject_dist.keys():
            one_subject_dist[main_subject] = 0
        one_subject_dist[main_subject] += 1
        selected_tweets.append(diff(tweet, stimuli.keys()))
        targets.append(main_subject)

    if subject_count not in subject_dist.keys():
        subject_dist[subject_count] = 0
    if tag_count <= 5:
        subject_dist[subject_count] += 1

print(len(tweets))
print(subject_dist)
print(one_subject_dist)

vector_path = 'outputs/w2v/w2v_model_minCount10_vectors.csv'
vocab_vectors = np.loadtxt(vector_path, delimiter=' ', dtype='str', comments=None, skiprows=1)
wv_vocab = vocab_vectors[:, 0]
wv_vectors = vocab_vectors[:, 1:].astype('float')

vector_path = 'outputs/word2vec_10_20_50_noPrune.csv'
vocab_vectors = np.loadtxt(vector_path, delimiter=' ', dtype='str', comments=None, skiprows=1)
nv_vocab = vocab_vectors[:, 0]
nv_vectors = vocab_vectors[:, 1:].astype('float')


def embed_tweet(tweet):
    index_of_mv_tokens = np.where(np.isin(wv_vocab, tweet))
    index_of_nv_tokens = np.where(np.isin(nv_vocab, tweet))
    vectors_of_mv_tokens = wv_vectors[index_of_mv_tokens]
    vectors_of_nv_tokens = nv_vectors[index_of_nv_tokens]
    # tweet_embedding = np.concatenate([
    #     np.mean(vectors_of_mv_tokens, axis=0),
    #     np.mean(vectors_of_nv_tokens, axis=0)
    # ])
    tweet_embedding = np.mean(vectors_of_mv_tokens, axis=0)
    return tweet_embedding


tweet_vectors = []
targets = np.array(targets)
cntr = 0
for tweet in selected_tweets:
    cntr += 1
    if cntr % 1000 == 0:
        print('%' + str(int(cntr / len(selected_tweets) * 100)))
    tweet_vector = embed_tweet(tweet)
    tweet_vectors.append(tweet_vector)
# tweet_vectors = v_tweet_embedder(tweets, object)

tweet_vectors_np = np.array(tweet_vectors)
print(tweet_vectors_np)

print(tweet_vectors_np.shape)
t = targets.reshape((targets.shape[0], 1))
print(t.shape)
v = np.concatenate((tweet_vectors_np, t), axis=1)
print(v)

np.savetxt('outputs/subject_classification_dataset_2.txt', v, fmt='%s', delimiter=',')
