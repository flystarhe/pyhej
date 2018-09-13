import editdistance
from pyhej.nlp.trie import Trie
from pyhej.nlp.string import normalize


def tool_ngram_skip(text, skip=1):
    tmp = '?' * skip
    return [i+j for i,j in zip(tmp+text, text+tmp)]


def tool_similarity_ngram(lstr, rstr):
    lset = set(tool_ngram_skip(lstr))
    rset = set(tool_ngram_skip(rstr))
    return 2 * len(lset & rset) / (len(lset) + len(rset))


def tool_similarity_editdistance(lstr, rstr):
    return 1 - editdistance.eval(lstr, rstr) / max(1, len(lstr), len(rstr))


class EditDistance(object):
    def __init__(self, words):
        self.words = list(words)

    def near(self, word, n=5):
        temp_list = []
        for item in self.words:
            temp = self.dist(word, item)
            temp_list.append((item, temp[1]))
        temp_list = sorted(temp_list, key=lambda x: x[1])
        if len(temp_list) < n:
            return temp_list
        return temp_list[-n:]

    def dist(self, str1, str2):
        num_dist = editdistance.eval(str1, str2)
        num_sim = 1 - num_dist / max(len(str1), len(str2))
        return (num_dist, num_sim)

    def topn(self, n, *args):
        temp_list = []
        for arg in args:
            temp_list.extend(arg)
        temp_list = sorted(temp_list, key=lambda x: x[1])
        keys, vals = [], []
        while 1:
            if len(keys) < n and len(temp_list) > 0:
                key, val = temp_list.pop()
                if key not in keys:
                    keys.append(key)
                    vals.append(val)
            else:
                break
        return (keys, vals)


class Similarity(object):
    starts = Trie()
    ends = Trie()
    data = []

    def __init__(self, starts, ends):
        for word, label in starts:
            self.starts.add(word, label)
        for word, label in ends:
            self.ends.add_reverse(word, label)

    def set(self, data):
        for name, prefix, core, suffix in data:
            tmp = self.starts.find(prefix)
            if tmp is not None:
                tag_s = tmp
            else:
                tag_s = prefix
            tmp = self.ends.find_reverse(suffix)
            if tmp is not None:
                tag_e = tmp
            else:
                tag_e = suffix
            self.data.append((name, prefix, core, suffix, tag_e, tag_s))

    def split(self, word):
        word = normalize(word)
        pos_s, tag_s = self.starts.match(word)
        pos_e, tag_e = self.ends.match_reverse(word)
        if pos_s is None:
            pos_s = 0
            tag_s = '_'
            prefix = '_'
        else:
            prefix = word[:pos_s]
        if pos_e is None:
            pos_e = 0
            tag_e = '_'
            suffix = '_'
        else:
            suffix = word[-pos_e:]
        return (prefix, word[pos_s:len(word) - pos_e], suffix, tag_e, tag_s)

    def similarity(self, word, word_):
        num_dist = editdistance.eval(word, word_)
        return 1 - num_dist / max(len(word), len(word_))

    def dist_prefix(self, word, tag, word_, tag_):
        if word == word_:
            return 1.0
        if tag == tag_:
            return 1.0
        if word == '_':
            return 0.8
        if word_ == '_':
            return 0.8
        return self.similarity(word, word_)

    def dist_core(self, word, word_):
        return self.similarity(word, word_)

    def dist_suffix(self, word, tag, word_, tag_):
        if word == word_:
            return 1.0
        if tag == tag_:
            return 1.0
        if word == '_':
            return 0.9
        if word_ == '_':
            return 0.9
        return self.similarity(word, word_)

    def score(self, prefix, core, suffix, tag_e, tag_s, n=5):
        temp_list = []
        for name_, prefix_, core_, suffix_, tag_e_, tag_s_ in self.data:
            score_s = self.dist_prefix(prefix, tag_s, prefix_, tag_s_)
            score_m = self.dist_core(core, core_)
            score_e = self.dist_suffix(suffix, tag_e, suffix_, tag_e_)
            temp_list.append((name_, score_s*score_m*score_e))
        temp_list = sorted(temp_list, key=lambda x: x[1])
        if len(temp_list) < n:
            return temp_list
        return temp_list[-n:]

    def near(self, word, n=5):
        temp_list = []
        prefix, core, suffix, tag_e, tag_s = self.split(word)
        temp_list.extend(self.score(prefix, core, suffix, tag_e, tag_s, n))
        if prefix != '_':
            temp_list.extend(self.score('_', prefix + core, suffix, tag_e, '_', n))
        if suffix != '_' and core == '':
            temp_list.extend(self.score(prefix, core + suffix, '_', '_', tag_s, n))
        return temp_list

    def best(self, n, data):
        temp_list = sorted(data, key=lambda x: x[1])
        keys, vals = [], []
        while 1:
            if len(keys) < n and len(temp_list) > 0:
                key, val = temp_list.pop()
                if key not in keys:
                    keys.append(key)
                    vals.append(val)
            else:
                break
        return (keys, vals)

    def topn(self, n, *args):
        temp_list = []
        for arg in args:
            temp_list.extend(arg)
        temp_list = sorted(temp_list, key=lambda x: x[1])
        keys, vals = [], []
        while 1:
            if len(keys) < n and len(temp_list) > 0:
                key, val = temp_list.pop()
                if key not in keys:
                    keys.append(key)
                    vals.append(val)
            else:
                break
        return (keys, vals)