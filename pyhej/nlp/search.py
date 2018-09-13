import codecs
from pyhej.nlp.trie import Trie


class Faster(object):
    '''docstring for Faster
    '''
    def __init__(self, data):
        self.trie = Trie()
        for word, label in data:
            self.trie.add(word, label)

    def add_in_file(self, fpath):
        with codecs.open(fpath, 'r', 'utf-8') as reader:
            for line in reader.readlines():
                temp = line.strip().split(',')
                self.trie.add(temp[0], temp[1])

    def add_in_list(self, data):
        for word, label in data:
            self.trie.add(word, label)

    def add(self, word, tag):
        self.trie.add(word, tag)

    def delete(self, word):
        self.trie.delete(word)

    def find(self, word):
        return self.trie.find(word)