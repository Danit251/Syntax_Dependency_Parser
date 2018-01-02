from collections import namedtuple, defaultdict

from nltk.corpus import dependency_treebank
import numpy as np
from scipy.sparse import dok_matrix
from copy import deepcopy
import Node
import Edge
import time

corpus_sentences = dependency_treebank.parsed_sents()

training_size = round(len(corpus_sentences) * 0.9)
training_set = corpus_sentences[:training_size]
test_set = corpus_sentences[training_size:]

NUM_ITER = 2

words_deps = {}
words_dict = {}
tags_dict = {}
feature_vec_size = 0

additional_features = True

Arc = namedtuple('Arc', ('tail', 'weight', 'head'))


# --------------------------- helper Functions --------------------------------



# --------------------------- prepare data ------------------------------------
def set_dicts(corpus):
    global words_dict
    global tags_dict
    words_dict = {None: 0}
    tags_dict = {'TOP': 0}
    for tree in corpus:
        for i in range(1, len(tree.nodes)):
            word = tree.nodes[i]["word"]
            tag = tree.nodes[i]["tag"]
            if word not in words_dict and word is not None:
                words_dict[word] = len(words_dict)
            if tag not in tags_dict and tag is not None:
                tags_dict[tag] = len(tags_dict)


def feature_function(node1, node2):
    feature_size = len(words_dict) ** 2 + len(tags_dict) ** 2
    if additional_features:
        feature_size = feature_size + 4
    feature_vec = dok_matrix((feature_size, 1))
    # word_feature_ind = -1

    if node1['word'] in words_dict:
        word1_ind = words_dict[node1['word']]
        if node2['word'] in words_dict:
            word2_ind = words_dict[node2['word']]
            word_feature_ind = word1_ind * len(words_dict) + word2_ind
            feature_vec[word_feature_ind, 0] = 1

    if node1['tag'] in tags_dict:
        tag1_ind = tags_dict[node1['tag']]
        if node2['tag'] in tags_dict:
            tag2_ind = tags_dict[node2['tag']]
            tag_feature_ind = len(words_dict) ** 2 + tag1_ind * len(
                tags_dict) + tag2_ind
            feature_vec[tag_feature_ind, 0] = 1

    # part e:
    if additional_features:
        add1 = node1['address']
        add2 = node2['address']
        if (add2 - add1) < 4 and (add2-add1) > 0:
            feature_vec[-(5-add2-add1)] = 1

    return feature_vec


# ------------------------------- MST ----------------------------------------

def build_tree_from_sent(teta, tree):
    # TODO minus?
    edges = []
    for i in range(1, len(tree.nodes)):
        node = tree.nodes[i]
        weight = calc_score(tree.nodes[0], node, teta)
        edges.append(Arc(i, weight, 0))
        for j in range(1, len(tree.nodes)):
            if j != i:
                weight = calc_score(tree.nodes[j], node, teta)
                edges.append(Arc(i, weight, j))
    return edges


def mst(arcs, sink):
    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[sink] = sink
    while True:
        min_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            if arc.tail == sink:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if tail_rep not in min_arc_by_tail_rep or min_arc_by_tail_rep[
                tail_rep].weight < arc.weight:
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, sink)
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, sink)
        good_arcs.extend(
            min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        quotient_map = {
            node: cycle_rep if node_rep in cycle_rep_set else node_rep for
            node, node_rep in quotient_map.items()}


def find_cycle(successor, sink):
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None


def spanning_arborescence(arcs, sink):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail


def calc_score(node1, node2, teta):
    vec = feature_function(node1, node2)
    current_score = 0
    for item in vec.items():
        current_score += teta[item[0]]
    return current_score


# ----------------------------- part c ----------------------------------------

def perceptron(feature_size, num_iter):
    if additional_features:
        feature_size = feature_size + 4
    curr_teta = dok_matrix((feature_size, 1))
    sum_teta = curr_teta
    # shuffled_training = deepcopy(training_set[0:4000])
    shuffled_training = deepcopy(training_set[0:500])

    for r in range(num_iter):
        np.random.shuffle(shuffled_training)
        for i, tree in enumerate(shuffled_training):
            mst_edges = calc_tree(curr_teta, tree)
            right_edges = calc_right_tree(tree)
            sum_mst = -1 * sum_features_edges(tree, mst_edges, feature_size)
            sum_right = sum_features_edges(tree, right_edges, feature_size)
            curr_teta += sum_mst + sum_right
            sum_teta += curr_teta

    res = sum_teta / (num_iter * len(shuffled_training))

    return res


# activate MST algorithm
def calc_tree(teta, tree):
    edges = build_tree_from_sent(teta, tree)
    return mst(edges, 0)


def update_edges_set(tree, dep_num, edges_set, edge_ind, node1):
    curr_edge = Arc(edge_ind, dep_num, node1['address'])
    edges_set.add(curr_edge)
    edge_ind += 1
    return edges_set, edge_ind


def calc_right_tree(tree):
    edges_set = {}
    edge_ind = 0
    for i in tree.nodes:
        word = tree.nodes[i]["word"]
        deps = tree.nodes[i]["deps"]
        if word is None:
            for dep_num in deps['ROOT']:
                edges_set[edge_ind] = Arc(dep_num, 0, i)
                edge_ind += 1
        else:
            for dep_num in deps['']:
                edges_set[edge_ind] = Arc(dep_num, 0, i)
                edge_ind += 1
    return edges_set


def sum_features_edges(tree, edges_set, feature_size):
    # TODO REMOVE DICT
    dict = {}
    edges_sum = dok_matrix((feature_size, 1))
    for ind in edges_set:
        edge = edges_set[ind]
        if edge.head in dict \
                and dict[edge.head] == edge.tail:
            continue
        out_node = tree.nodes[edge.head]
        in_node = tree.nodes[edge.tail]
        edges_sum += feature_function(out_node, in_node)
        dict[edge.head] = edge.tail
    return edges_sum


# ----------------------------- part d ----------------------------------------

def test(teta):
    num_edges = 0
    num_right_edges = 0
    for tree in test_set:
        mst_edges = calc_tree(teta, tree)
        right_edges = calc_right_tree(tree)
        num_edges += len(right_edges)

        for i in right_edges:
            for j in mst_edges:
                right_edge = right_edges[i]
                mst_edge = mst_edges[j]
                if (right_edge.head == mst_edge.head) \
                        and (right_edge.tail == mst_edge.tail):
                    num_right_edges += 1
                    break

    return num_right_edges / num_edges


# ------------------------------ Main -----------------------------------------

def main():
    set_dicts(training_set)
    feature_size = len(words_dict) ** 2 + len(tags_dict) ** 2
    global additional_features
    additional_features = True
    # additional_features = False

    # todo delete
    start = time.time()
    print("additional features:", additional_features)

    # training:
    res = perceptron(feature_size, num_iter=NUM_ITER)

    end = time.time()
    print("Training time:", end - start)

    # evaluation:
    print("Success:", test(res))

    end2 = time.time()
    print("Evaluation time", end2 - end)


if __name__ == '__main__':
    main()
