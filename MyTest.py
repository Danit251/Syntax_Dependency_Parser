import ex3
from nltk.corpus import dependency_treebank
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import Node, Edge

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


def test_b_and_e():
    edges_set = ex3.calc_right_tree(training_set[0])
    sentence = ex3.create_sentence(training_set[0])
    print(sentence)
    # for edge in edges_set:
    #     print(edge.out_node.word, edge.in_node.word)
    #     print(ex3.feature_function(edge.out_node, edge.in_node, sentence))

    sentence= []
    sentence.append(('The', 'NNP'))
    sentence.append(('good', 'NNP'))
    sentence.append(('director', 'NNP'))
    sentence.append(('board', 'NNP'))
    sentence.append(('as', 'NNP'))
    sentence.append(('Nov.', 'NNP'))
    sentence = np.array(sentence)

    node1 = Node.Node(sentence[1][0], 1, sentence[0][1])
    node2 = Node.Node(sentence[2][0], 1, sentence[5][1])

    print(ex3.feature_function(node1, node2, sentence))


def test_c():
    teta = dok_matrix(
        (len(ex3.words_dict) ** 2 + len(ex3.tags_dict) ** 2 + 4, 1))

    print("teta", teta)
    tree_score = ex3.tree_score(training_set[0], teta)
    print(tree_score)


def main():
    ex3.set_dicts(training_set)
    feature_size = len(ex3.words_dict) ** 2 + len(ex3.tags_dict) ** 2 + 4
    # test_b_and_e()



    # 126699745
    # print("")
    # print("num", ex3.sum_features_edges(edges_set, sentence, feature_size))

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
