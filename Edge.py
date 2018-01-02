
class Edge:

    def __init__(self, edge_id, out_node, in_node, weight):
        self.id = edge_id
        self.weight = weight
        self.in_node = in_node
        self.out_node = out_node
        # self.origin_in_node = in_node
        # self.origin_out_node = out_node
