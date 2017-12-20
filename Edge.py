
class Edge:
    id = -1
    weight = -1
    origin_weight = -1
    in_node = None
    out_node = None

    def __init__(self, id, in_node, out_node, weight):
        self.id = id
        self.weight = weight
        self.origin_weight = weight
        self.in_node = in_node
        self.out_node = out_node
