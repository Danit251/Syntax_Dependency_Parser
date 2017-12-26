import ex3
from nltk.corpus import dependency_treebank
from scipy.sparse import dok_matrix, csr_matrix

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
    # test_b()
    test_c()


if __name__ == "__main__":
    main()
