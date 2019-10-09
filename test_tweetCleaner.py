from Utils.TweetCleaner import TweetCleaner
from Utils.WordNormalizer import WordNormalizer

tweetCleaner = TweetCleaner()
file_name = 'inputs/tweets_likeCount.tsv'
# hashtags_file_name = file_name.replace('.tsv', '').replace('.csv', '') + '_hashtags.txt'
# words_file_name = file_name.replace('.tsv', '').replace('.csv', '') + '_words.txt'
tokens_file_name = file_name.replace('.tsv', '').replace('.csv', '') + '_tokens.txt'

nor = WordNormalizer()
nor.read_resource()

with open(file_name, 'r') as input_f, open(tokens_file_name, 'w+') as tokens_f:
    cntr = 0
    while True:
        cntr += 1
        if cntr % 10000 == 0:
            print(cntr)
        line = input_f.readline()
        if not line:
            break
        if cntr == 113641:
            continue
        tweet = '\t'.join(line.split('\t')[1:-1])
        like_count = line.split('\t')[-1]
        normalized_tweet = nor.normalize(tweet)
        # words, hashtags = tweetCleaner.clean(normalized_tweet)
        tokens = tweetCleaner.clean(normalized_tweet)
        tokens_f.write(' '.join(tokens) + '\t' + like_count)