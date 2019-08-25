from Twitter.TweetCleaner import TweetCleaner

tweetCleaner = TweetCleaner()
file_name = 'tweets_100samples.txt'
hashtags_file_name = file_name.replace('.txt', '').replace('.csv', '') + '_words.txt'
words_file_name = file_name.replace('.txt', '').replace('.csv', '') + '_hashtags.txt'

with open(file_name, 'r') as input_f, open(hashtags_file_name, 'w+') as hashtags_f, open(words_file_name,
                                                                                         'w+') as words_f:
    while True:
        tweet = input_f.readline()
        if not tweet:
            break
        words, hashtags = tweetCleaner.clean(tweet)
        words_f.write(' '.join(words)+'\n')
        hashtags_f.write(' '.join(hashtags)+'\n')