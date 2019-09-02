import networkx as nx

from Utils.ResponseInPack import ResponseInPack
from Utils.WordNormalizer import WordNormalizer


class WAN:
    def __init__(self, path):
        self.graph = nx.Graph()
        self.nodes = {}
        with open(path) as f:
            all_responses = f.read()
            responses = [ResponseInPack(line) for line in all_responses.split('\n')]

        wn = WordNormalizer()
        wn.read_resource()

        for response in responses:
            source = wn.normalize(response.stimulus).replace(' ', '_')
            target = wn.normalize(response.response).replace(' ', '_')
            pack_id = int(response.pack_id)
            if source != '' and target != '' and source != target:
                self.__add_edge(source, target)
                self.__add_node(source, -1)
                self.__add_node(target, pack_id)

    def prune_edge(self, min_weight):
        pruned_wan = nx.Graph()
        pruned_edges = [(u, v, d['weight']) for (u, v, d) in self.graph.edges(data=True) if d['weight'] >= min_weight]
        pruned_wan.add_weighted_edges_from(pruned_edges)
        self.graph = pruned_wan

    def prune_node(self, min_freq):
        pruned_wan = nx.Graph()
        pruned_nodes = [u for u in self.nodes.keys() if len(self.nodes[u]) >= min_freq]
        pruned_edges = [(u, v, d['weight']) for (u, v, d) in self.graph.edges(data=True) if
                        (u in pruned_nodes and v in pruned_nodes)]
        pruned_wan.add_weighted_edges_from(pruned_edges)
        self.graph = pruned_wan

    def reverse_weight(self):
        for edge in self.graph.edges:
            self.graph.edges[edge]['weight'] = 1 / self.graph.edges[edge]['weight']

    def __add_edge(self, source, target):
        if not self.graph.has_edge(source, target):
            self.graph.add_edge(source, target, weight=0)
        self.graph.edges[source, target]['weight'] += 1

    def __add_node(self, source, pack_id):
        if source not in self.nodes.keys():
            self.nodes[source] = set()
        self.nodes[source].add(pack_id)
