class Trie(object):
    def __init__(self):
        self.root = {}
        self.END = '/'

    def add(self, word, tag=None):
        node = self.root
        for c in word:
            node = node.setdefault(c, {})
        node[self.END] = tag

    def add_reverse(self, word, tag=None):
        self.add(word[::-1], tag)

    def find(self, word):
        node = self.root
        for c in word:
            if c not in node:
                return None
            node = node[c]
        return node.get(self.END)

    def find_reverse(self, word):
        return self.find(word[::-1])

    def match(self, word):
        res = (None, None)
        node = self.root
        for i, c in enumerate(word):
            if c not in node:
                break
            node = node[c]
            if self.END in node:
                res = (i + 1, node.get(self.END))
        return res

    def match_reverse(self, word):
        return self.match(word[::-1])

    def delete(self, word):
        node = self.root
        for c in word:
            if c not in node:
                return None
            node = node[c]
        return node.pop(self.END)

    def delete_reverse(self, word):
        return self.delete(word[::-1])