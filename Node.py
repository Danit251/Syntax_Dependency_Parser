
class Node:
    word = None
    index = None
    incoming_edges = []
    outgoing_edges = []

    def __init__(self, word, index, incoming_edges, outgoing_edges):
        self.word = word
        self.index = index
        self.incoming_edges = incoming_edges
        self.outgoing_edges = outgoing_edges

    def get_max_incoming(self):
        max_weight = -1
        max_edge = None
        for edge in self.incoming_edges:
            if max_weight < edge[2]:
                max_weight = edge[2]
                max_edge = edge

        return max_edge

    def update_weights(self):
        max_edge = self.get_max_incoming()
        max_weight = max_edge[2]
        for edge in self.incoming_edges:
            if edge != max_edge:
                edge[2] -= max_weight