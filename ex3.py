from nltk.corpus import dependency_treebank
import numpy as np
from copy import deepcopy

corpus_sentences = dependency_treebank.parsed_sents()

training_size = round(len(corpus_sentences) * 0.9)
training_set = corpus_sentences[:training_size]
test_set = corpus_sentences[training_size:]

NUM_ITER = 2


def perceptron(feature_size, num_iter, feature_func):
    teta = np.zeros(feature_size)
    shuffled_training = deepcopy(training_set)
    np.random.shuffle(shuffled_training)
    corpus_size = len(corpus_sentences)
    for r in range(num_iter):
        for i, tree in enumerate(shuffled_training):
            sentence = create_sentence(tree)
            # This should do the MST - we think so
            tree_result = tree_score(sentence)
            cur_teta = (r - 1) * corpus_size + i
            teta[cur_teta] = teta[cur_teta - 1] + \
                             calc_tree_features(tree, sentence) \
                                - calc_tree_features(tree_result, sentence)
    return np.sum(teta)/(num_iter*corpus_size)


def tree_score(sentence):
    return 0


def calc_tree_features(tree, sentence):
    pass


def create_sentence(tree):
    sentence = []
    for i in range(len(tree.nodes)):
        sentence.append((tree.nodes[i]["word"], tree.nodes[i]["tag"]))
    return sentence

print(create_sentence(training_set[0]))
print(create_sentence(training_set[1]))