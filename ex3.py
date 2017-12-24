from nltk.corpus import dependency_treebank
import numpy as np
from copy import deepcopy
import Node, Edge

corpus_sentences = dependency_treebank.parsed_sents()

training_size = round(len(corpus_sentences) * 0.9)
training_set = corpus_sentences[:training_size]
test_set = corpus_sentences[training_size:]

NUM_ITER = 2

words_deps = {}
words_dict = {}
tags_dict = {}


# Helper functions

# for word - enter 0, for tag- enter 1
def create_sentence(tree):
    sentence = []

    # for ROOT
    # sentence.append(("ROOT", "ROOT"))

    for i in range(1, len(tree.nodes)):
        sentence.append((tree.nodes[i]["word"], tree.nodes[i]["tag"]))

    return np.array(sentence)


def find_in_sentence(sentence, value, word_or_tag=0):
    for i, node in enumerate(sentence):
        if node[word_or_tag] == value:
            return i
    return -1


def set_dicts(corpus):
    for tree in corpus:
        for i in range(0, len(tree.nodes)):
            word = tree.nodes[i]["word"]
            tag = tree.nodes[i]["tag"]
            deps = tree.nodes[i]["deps"]
            define_weights(word, deps, tree)
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            if tag not in tags_dict:
                tags_dict[tag] = len(tags_dict)


def define_weights(word, deps, tree):
    if word is None:
        for dep_num in deps['ROOT']:
            dep = tree.nodes[dep_num]["word"]
            if 'ROOT' in words_deps:
                if dep in words_deps['ROOT']:
                    words_deps['ROOT'][dep] += 1
                else:
                    words_deps['ROOT'][dep] = 1
            else:
                words_deps['ROOT'] = {dep: 1}
    else:
        for dep_num in deps['']:
            dep = tree.nodes[dep_num]["word"]
            if word in words_deps:
                if dep in words_deps[word]:
                    words_deps[word][dep] += 1
                else:
                    words_deps[word][dep] = 1
            else:
                words_deps[word] = {dep: 1}


# part b
def feature_function(node1, node2, sentence):
    word1_ind = words_dict[node1[0]]
    word2_ind = words_dict[node2[0]]
    tag1_ind = tags_dict[node1[1]]
    tag2_ind = tags_dict[node2[1]]

    feature_vec = np.zeros(len(words_dict) ** 2 + len(tags_dict) ** 2 + 4)
    word_feature_ind = tag_feature_ind = -1

    if word2_ind != -1 and word1_ind != -1:
        word_feature_ind = word1_ind * len(words_dict) + word2_ind
        feature_vec[word_feature_ind] = 1

    if tag1_ind != -1 and tag2_ind != -1:
        tag_feature_ind = len(words_dict) ** 2 + tag1_ind * len(
            tags_dict) + tag2_ind
        feature_vec[tag_feature_ind] = 1

    # part e:
    if word_feature_ind != -1:
        word1_ind = find_in_sentence(sentence, node1)
        word2_ind = find_in_sentence(sentence, node2)

        if word2_ind - word1_ind == 1:
            feature_vec[-1] = 1
        else:
            if word1_ind - word2_ind == 1:
                feature_vec[-2] = 1

    if tag_feature_ind != -1:
        tag1_ind = find_in_sentence(sentence, node1, False)
        tag2_ind = find_in_sentence(sentence, node2, False)

        if tag2_ind - tag1_ind == 1:
            feature_vec[-3] = 1
        else:
            if tag1_ind - tag2_ind == 1:
                feature_vec[-4] = 1

    return feature_vec


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


def calc_score(node1, node2, teta):
    word1_ind = words_dict[node1[0]]
    word2_ind = words_dict[node2[0]]
    tag1_ind = tags_dict[node1[1]]
    tag2_ind = tags_dict[node2[1]]

    sum = 0
    word_feature_ind = tag_feature_ind = -1

    if word2_ind != -1 and word1_ind != -1:
        word_feature_ind = word1_ind * len(words_dict) + word2_ind
        sum += teta[word_feature_ind]

    if tag1_ind != -1 and tag2_ind != -1:
        tag_feature_ind = len(words_dict) ** 2 + tag1_ind * len(
            tags_dict) + tag2_ind
        sum += teta[tag_feature_ind]

    if word_feature_ind != -1:
        if word2_ind - word1_ind == 1:
            sum += teta[-1]
        else:
            if word1_ind - word2_ind == 1:
                sum += teta[-2]
    if tag_feature_ind != -1:
        if tag2_ind - tag1_ind == 1:
            sum += teta[-3]
        elif tag1_ind - tag2_ind == 1:
            sum += teta[-4]
    return sum


def tree_score(tree, teta):
    sum_score = 0
    # for i in range(1, len(tree.nodes)):
    #
    #     for j in range(1, len(tree.nodes)):

    return 0


def build_tree_from_sent(sentence):
    nodes = []
    edges = {}
    root_node = Node.Node("", 0)
    index_node = 1
    index_edge = 1

    # Creates all the nodes of the graph
    for pair in sentence:
        word = pair[0]
        new_node = Node.Node(word, index_node)
        nodes.append(new_node)
        index_node += 1

    # Creates all the edges of all the nodes except the root node
    for fst_node in nodes:
        for scd_node in nodes:
            weight = 0
            if fst_node.id == scd_node.id:
                continue
            if fst_node.word in words_deps:
                if scd_node.word in words_deps[fst_node.word]:
                    weight = words_deps[fst_node.word][scd_node.word]
            new_edge = Edge.Edge(index_edge, fst_node, scd_node, weight)
            edges[new_edge.id] = new_edge
            fst_node.add_outgoing_edge(new_edge)
            scd_node.add_incoming_edge(new_edge)
            index_edge += 1

    # Creates all the outgoing edges of the root node
    for kodkod in nodes:
        weight = 0
        if kodkod.word in words_deps["ROOT"]:
            weight = words_deps["ROOT"][kodkod.word]
        new_edge = Edge.Edge(index_edge, root_node, kodkod, weight)
        edges[new_edge.id] = new_edge
        root_node.add_outgoing_edge(new_edge)
        kodkod.add_incoming_edge(new_edge)
        index_edge += 1

    return nodes, edges


# Gets all the nodes of the graph(without the root) and find the MST
def mst(nodes, edges):
    num_nodes = len(nodes)
    # Each element: edge.id
    best_in_edge = []
    # Each element: edge.id : {edges.id}
    kicks_out = {}
    cur_edges = []
    while nodes:
        cur_node = nodes.pop()
        max_edge = cur_node.get_max_incoming()
        best_in_edge.append(max_edge.id)
        cur_edges.append(max_edge)
        cycle = is_cycle(cur_edges)
        if cycle:
            nodes_in_cycle, edges_in_cycle = cycle
            edges_in_cycle = set(edges_in_cycle)
            num_nodes += 1
            # Updates the kicks_out nodes array
            for node in nodes_in_cycle:
                # Updates weight of the vertexes in the cycle and update
                max_tzela = node.get_max_incoming()
                node.update_weights()
                for incoming_edge in node.incoming_edges:
                    if incoming_edge.id != max_tzela.id:
                        if incoming_edge.id not in kicks_out:
                            kicks_out[incoming_edge.id] = []
                        kicks_out[incoming_edge.id].append(max_tzela.id)

            new_node = create_united_nodes(nodes_in_cycle, num_nodes)
            nodes.append(new_node)
            cur_edges = [item for item in cur_edges if
                         item not in edges_in_cycle]

    # Kicks the bad edges in the reverse way
    best_in_edge.reverse()
    for edge_id in best_in_edge:
        if edge_id in kicks_out:
            for i, tzela in enumerate(best_in_edge):
                if tzela in kicks_out[edge_id]:
                    best_in_edge[i] = edge_id
    all_weights = 0
    for best_edge in best_in_edge:
        all_weights += edges[best_edge].origin_weight
    return all_weights


def is_cycle(edges):
    path_id = set()
    path_edges = []
    path = set()
    visited = set()

    def visit(cur_edge, all_edges):
        if cur_edge.in_node.id in visited:
            return None

        vertex = cur_edge.in_node
        visited.add(vertex.id)
        path_id.add(vertex.id)
        path.add(vertex)
        path_edges.append(cur_edge)

        for edge in vertex.outgoing_edges:
            if edge not in all_edges:
                continue
            path_edges.append(edge)
            t = visit(edge, all_edges)
            if edge.in_node.id in path_id or t:
                return path, path_edges
            path_edges.remove(edge)

        path_id.remove(vertex.id)
        path.remove(vertex)
        path_edges.remove(cur_edge)
        return None

    for my_edge in edges:
        # if my_edge.out_node.id == 0:
        #     continue
        path_id.add(my_edge.out_node.id)
        path.add(my_edge.out_node)
        cycle = visit(my_edge, edges)
        if cycle:
            return cycle
        path_id.remove(my_edge.out_node.id)
        path.remove(my_edge.out_node)
    return None


def create_united_nodes(nodes_to_union, index):
    new_node = Node.Node("", index)
    for node in nodes_to_union:

        # Adds and updates all the incoming edges to the united node
        for incoming_edge in node.incoming_edges:
            incoming_edge.in_node = new_node
            # Does not add edges between the united nodes
            if incoming_edge.out_node not in nodes_to_union and incoming_edge.out_node is not new_node:
                new_node.add_incoming_edge(incoming_edge)

        # Adds and updates all the outgoing edges to the united node
        for outgoing_edge in node.outgoing_edges:
            outgoing_edge.out_node = new_node
            # Does not add edges between the united nodes
            if outgoing_edge.in_node not in nodes_to_union and incoming_edge.in_node is not new_node:
                new_node.add_outgoing_edge(outgoing_edge)
    return new_node


def calc_tree_features(tree, sentence):
    pass


def main():
    set_dicts(training_set)
    sen = create_sentence(training_set[7])
    sen1 = create_sentence(training_set[0])
    g, e = build_tree_from_sent(sen)
    g1, e1 = build_tree_from_sent(sen1)
    print(mst(g, e))
    print(mst(g1, e1))

if __name__ == "__main__":
    main()
