from nltk.corpus import dependency_treebank
import numpy as np
from copy import deepcopy
import Node, Edge

corpus_sentences = dependency_treebank.parsed_sents()

training_size = round(len(corpus_sentences) * 0.9)
training_set = corpus_sentences[:training_size]
test_set = corpus_sentences[training_size:]

NUM_ITER = 2


# Helper functions

# for word - enter 0, for tag- enter 1
def create_sentence(tree):
    sentence = []

    # for ROOT
    sentence.append(("ROOT", "ROOT"))

    for i in range(1, len(tree.nodes)):
        sentence.append((tree.nodes[i]["word"], tree.nodes[i]["tag"]))

    return np.array(sentence)


def find_in_sentence(sentence, value, word_or_tag=0):
    for i, node in enumerate(sentence):
        if node[word_or_tag] == value:
            return i
    return -1



# part b
def feature_function(node1, node2, sentence):
    length = len(sentence)

    word1_ind = find_in_sentence(sentence, node1[0])
    word2_ind = find_in_sentence(sentence, node2[0])
    tag1_ind = find_in_sentence(sentence, node1[1], word_or_tag=1)
    tag2_ind = find_in_sentence(sentence, node2[1], word_or_tag=1)

    feature_vec = np.zeros(length ** 2 + length ** 2 + 4)
    word_feature_ind = tag_feature_ind = -1

    if word2_ind != -1 and word1_ind != -1:
        word_feature_ind = word1_ind * length + word2_ind
        feature_vec[word_feature_ind] = 1

    if tag1_ind != -1 and tag2_ind != -1:
        tag_feature_ind = length ** 2 + tag1_ind * length + tag2_ind
        feature_vec[tag_feature_ind] = 1

    # part d:
    if word_feature_ind != -1:
        if word2_ind-word1_ind == 1:
            feature_vec[-1] = 1
        else:
            if word1_ind-word2_ind ==1:
                feature_vec[-2] = 1
    if tag_feature_ind != -1:
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


def tree_score(sentence):
    return 0


def mst(root, nodes):
    num_nodes = nodes.length
    # Each element: (node, edge)
    best_in_edge = {}
    # Each element: edge : {edges}
    kicks_out = {}
    cur_edges = []
    while nodes.length > 1:
        cur_node = nodes.pop()
        max_edge = cur_node.get_max_incoming()
        best_in_edge[cur_node.id] = max_edge.id
        cur_edges.append(max_edge)
        nodes_in_cycle = is_cycle(cur_edges)
        if nodes_in_cycle:
            num_nodes += 1
            # Updates the kicks_out nodes array
            for node in nodes_in_cycle:
                # Updates weight of the vertexes in the cycle and update
                node.update_weights()
                for incoming_edge in node.incoming_edges:
                    if incoming_edge.id != max_edge.id:
                        if incoming_edge.id not in kicks_out:
                            kicks_out[incoming_edge.id] = []
                        kicks_out[incoming_edge.id].append(max_edge.id)

            new_node = create_united_nodes(nodes_in_cycle, num_nodes)
            nodes.append(new_node)


def is_cycle(edges):
    path = set()
    visited = set()

    def visit(cur_edge):
        print("in visit")
        if cur_edge.in_node.id in visited:
            return None
        vertex = cur_edge.in_node
        visited.add(vertex.id)
        path.add(vertex.id)
        for edge in vertex.outgoing_edges:
            if edge.in_node.id in path or visit(edge):
                return path
        path.remove(vertex.id)
        return None

    for my_edge in edges:
        print(my_edge.id)
        path.add(my_edge.out_node.id)
        cycle_nodes = visit(my_edge)
        if cycle_nodes:
            return cycle_nodes
        path.remove(my_edge.out_node.id)
    return None


v1 = Node.Node("bla", 1)
v2 = Node.Node("bla", 2)
v3 = Node.Node("bla", 3)
v4 = Node.Node("bla", 4)
edge1 = Edge.Edge(11, v1, v2, 50)
edge2 = Edge.Edge(22, v2, v1, 50)
edge3 = Edge.Edge(33, v3, v1, 50)
edge4 = Edge.Edge(44, v3, v4, 50)
v1.add_outgoing_edge(edge1)
v2.add_incoming_edge(edge1)
v2.add_outgoing_edge(edge2)
v3.add_incoming_edge(edge2)
v3.add_outgoing_edge(edge3)
v3.add_outgoing_edge(edge4)
v1.add_incoming_edge(edge3)
v4.add_incoming_edge(edge4)
# edge1.out_node = v3
# print(v1.incoming_edges.pop().out_node.id)
# print(v2.incoming_edges.pop().out_node.id)
edges = [edge4, edge3, edge1, edge2]
print(is_cycle(edges))


def create_united_nodes(nodes_to_union, index):
    new_node = Node.Node("", index)
    for node in nodes_to_union:

        # Adds and updates all the incoming edges to the united node
        for incoming_edge in node.incoming_edges:
            incoming_edge.in_node = new_node
            # Does not add edges between the united nodes
            if incoming_edge.out_node not in nodes_to_union:
                new_node.add_incoming_edge(incoming_edge)

        # Adds and updates all the outgoing edges to the united node
        for outgoing_edge in node.outgoing_edge:
            outgoing_edge.in_node = new_node
            # Does not add edges between the united nodes
            if outgoing_edge.in_node not in nodes_to_union:
                new_node.add_outgoing_edge(outgoing_edge)
    return new_node


def calc_tree_features(tree, sentence):
    pass


def main():
    sentence = create_sentence(training_set[1])
    feature_vec = feature_function(sentence[0], sentence[1], sentence)

#
# if __name__ == "__main__":
#     main()
