from nltk.corpus import dependency_treebank
import numpy as np
from copy import deepcopy

corpus_sentences = dependency_treebank.parsed_sents()

training_size = round(len(corpus_sentences) * 0.9)
training_set = corpus_sentences[:training_size]
test_set = corpus_sentences[training_size:]

NUM_ITER = 2


# for word - enter 0, for tag- enter 1
def find_in_sentence(sentence, value, word_or_tag=0):
    for i, node in enumerate(sentence):
        if node[word_or_tag] == value:
            return i
    return -1


# part b
def feature_function(node1, node2, sentence):
    length = len(sentence)

    word1_ind = find_in_sentence(sentence, node1[0])
    word2_ind = find_in_sentence(sentence, node2[0]) + word1_ind * length
    tag1_ind = find_in_sentence(sentence, node1[1], word_or_tag=1)
    tag2_ind = find_in_sentence(sentence, node2[1], word_or_tag=1) \
               + length ** 2 + tag1_ind * length

    # 1 for root, marked as (None, 'TOP') todo fix later
    # additional 4 for part d, todo add later
    feature_vec = np.zeros(length ** 2 + length ** 2 + 4)
    feature_vec[word2_ind] = 1
    feature_vec[tag2_ind] = 1
    pass


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
    return np.sum(teta) / (num_iter * corpus_size)


def tree_score(sentence):
    return 0


def calc_tree_features(tree, sentence):
    pass


def create_sentence(tree):
    sentence = []
    for i in range(len(tree.nodes)):
        sentence.append((tree.nodes[i]["word"], tree.nodes[i]["tag"]))
    return np.array(sentence)


sentence = create_sentence(training_set[1])
feature_function(sentence[0], sentence[1], sentence)
# print(sentence[0][1])
# print(sentence)
# print(set(sentence[:,0]))
