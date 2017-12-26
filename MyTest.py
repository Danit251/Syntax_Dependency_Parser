import ex3
from nltk.corpus import dependency_treebank
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

corpus_sentences = dependency_treebank.parsed_sents()

training_size = round(len(corpus_sentences) * 0.9)
training_set = corpus_sentences[:training_size]
test_set = corpus_sentences[training_size:]


def test_b():
    tree = training_set[0]
    sentence = ex3.create_sentence(tree)
    node1 = tree.nodes[1]
    node2 = tree.nodes[2]
    f = ex3.feature_function(node1, node2, sentence)
    print(f)


def test_c():
    teta = dok_matrix(
        (len(ex3.words_dict) ** 2 + len(ex3.tags_dict) ** 2 + 4, 1))

    print("teta", teta)
    tree_score = ex3.tree_score(training_set[0], teta)
    print(tree_score)


def main():
    ex3.set_dicts(training_set)
    feature_size = len(ex3.words_dict) ** 2 + len(ex3.tags_dict) ** 2 + 4

    edges_set = ex3.calc_right_tree(training_set[0])
    sentence = ex3.create_sentence(training_set[0])
    # for edge in edges_set:
    #     print(edge.in_node.word, edge.out_node.word)
    #     print(ex3.feature_function(edge.out_node, edge.in_node, sentence))
    # print("")
    print("num", ex3.sum_features_edges(edges_set, sentence, feature_size))

    # teta = dok_matrix((4, 1))
    # teta2 = dok_matrix((4, 1))
    # arr = []
    # arr.append(teta)
    # arr.append(teta2)
    # teta[1,0] = 3
    # teta2[1,0] = 2
    # teta2[2,0] = 4
    # print(np.sum(arr))

    # test_b()
    # test_c()


if __name__ == "__main__":
    main()
