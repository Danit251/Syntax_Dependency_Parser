
class Edge:

    def __init__(self, edge_id, in_node, out_node, weight):
        self.id = edge_id
        self.weight = weight
        self.origin_weight = weight
        self.in_node = in_node
        self.out_node = out_node
