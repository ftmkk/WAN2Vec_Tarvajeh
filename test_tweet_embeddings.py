import numpy as np

vector_path = 'inputs/tweets_likeCount_tokens.txt'
tweet_lc = np.loadtxt(vector_path, delimiter='\t', dtype='str', comments=None, skiprows=1)
# tweet_lc = tweet_lc[:10, :]

vector_path = 'outputs/w2v/w2v_model_minCount10_vectors.csv'
vocab_vectors = np.loadtxt(vector_path, delimiter=' ', dtype='str', comments=None, skiprows=1)
vocab = vocab_vectors[:, 0]
vectors = vocab_vectors[:, 1:].astype('float')


def embed_tweet(tweet):
    tokens = tweet.split(' ')
    index_of_tokens = np.where(np.isin(vocab, tokens))
    vectors_of_tokens = vectors[index_of_tokens]
    tweet_embedding = np.mean(vectors_of_tokens, axis=0)
    return tweet_embedding


tweets = tweet_lc[:, 0]
targets = tweet_lc[:, 1].astype('int')

print(tweets)
print(targets)

tweet_vectors = []
cntr = 0
for tweet in tweets:
    cntr += 1
    if cntr % 1000 == 0:
        print('%'+str(int(cntr/len(tweets)*100)))
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

np.savetxt('outputs/v.txt', v, delimiter=',')
