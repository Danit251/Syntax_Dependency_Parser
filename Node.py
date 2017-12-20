
class Node:
    word = None
    id = -1
    incoming_edges = []
    outgoing_edges = []

    def __init__(self, word, id):
        self.word = word
        self.id = id

    def get_max_incoming(self):
        max_weight = -1
        max_edge = None
        for edge in self.incoming_edges:
            if max_weight < edge.weight:
                max_weight = edge.weight
                max_edge = edge
        return max_edge

    def update_weights(self):
        max_edge = self.get_max_incoming()
        max_weight = max_edge.weight
        for edge in self.incoming_edges:
            if edge.id != max_edge.id:
                edge.weight -= max_weight

    def add_incoming_edge(self, new_edge):
        self.incoming_edges.append(new_edge)

    def add_outgoing_edge(self, new_edge):
        self.outgoing_edges.append(new_edge)
