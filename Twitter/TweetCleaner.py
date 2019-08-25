import re

class TweetCleaner:
    url_pattern = r'\b((?:[a-z][\w-]+:' \
                  '(?:\/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)' \
                  '(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+' \
                  '(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\".,<>?«»“”‘’]))'
    url_regex = re.compile(url_pattern, re.IGNORECASE)
    tokens_pattern = r'((@?|\#?)[a-zA-Z0-9\']+)'
    tokens_regex = re.compile(tokens_pattern)

    def __remove_urls(self, t):
        return self.url_regex.sub('', t)

    def __words_and_hashtags(self, t):
        all_tokens = self.tokens_regex.findall(t)
        hashtags = []
        words = []
        for w in all_tokens:
            if w[1] == '#':
                hashtags.append(w[0])
            elif w[1] == '@':
                pass
            elif w[0] != 'RT':
                words.append(w[0])
        return words, hashtags

    def clean(self, t):
        return self.__words_and_hashtags(self.__remove_urls(t))