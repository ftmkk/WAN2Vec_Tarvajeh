class WordNormalizer:
    def __init__(self):
        self.delete_list = None
        self.replace_list = None

    def read_resource(self):
        with open('normalizer_resources/delete_list.txt') as f:
            self.delete_list = list(f.read().replace('\n', ''))
        with open('normalizer_resources/replace_list.txt') as f:
            self.replace_list = [
                (line.replace('"', '').split(',')[0], line.replace('"', '').split(',')[1])
                for line in f.read().split('\n')
            ]

    def normalize(self, word):
        normalized_word = word
        for char in self.delete_list:
            normalized_word = normalized_word.replace(char, '')
        for (char1, char2) in self.replace_list:
            normalized_word = normalized_word.replace(char1, char2)

        return normalized_word.strip()
