# https://github.com/orsinium/textdistance
#   pip install textdistance
import textdistance


def for_example():
    textdistance.hamming('test', 'text')
    # 1
    textdistance.hamming.distance('test', 'text')
    # 1
    textdistance.hamming.similarity('test', 'text')
    # 3
    textdistance.hamming.normalized_distance('test', 'text')
    # 0.25
    textdistance.hamming.normalized_similarity('test', 'text')
    # 0.75
    textdistance.Hamming(qval=2).distance('test', 'text')
    # 2