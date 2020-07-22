import networkx as nx
import numpy as np

# Build the example graph from the paper in networkx
n = 9
nodeList = list(range(1, 10))
edgeList = [(1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (4, 7), (5, 6), (5, 8), (6, 9), (7, 8), (8, 9)]
G = nx.Graph()
G.add_nodes_from(nodeList)
G.add_edges_from(edgeList)


# Test shattering functions
def uniquelist(list, idfunc=lambda x: x):
    """uniquify a list of lists"""

    seen = {}
    result = []

    for item in list:
        marker = idfunc(item)
        if marker not in seen:
            seen[marker] = 1
            result.append(item)
    return result


def vertexsetdistance(graph, vertex, part):
    """Define the distance between a vertex and a part of a partition"""
    return len([v for v in graph.adj[vertex] if v in part and v != vertex])


def set2setdistance(graph, part1, part2):
    """Define the vector valued distance between two parts in a partition"""
    return [vertexsetdistance(graph, v, part2) for v in part1]


def lexifyedges(graph):
    """Return edge set of a networkx graph with lexicographic ordering enforced"""
    # edges = list(graph.edges())
    lexify = lambda edge: (min(edge), max(edge))  # lexify a single edge
    lexEdges = sorted([lexify(edge) for edge in graph.edges()])
    return lexEdges

# Define partition class
class Partition:
    def __init__(self, partlist, graph, paintedvertices=[]):
        self.Graph = graph
        self.Parts = partlist
        self.PaintedVertices = paintedvertices

    def shattercheck(self, i, j):
        """Returns true if Vj shatters Vi"""
        return len(uniquelist([vertexsetdistance(self.Graph, v, self.Parts[j]) for v in self.Parts[i]])) != 1

    def shatter(self, i, j):
        """Return the shattering of Vi by Vj"""
        dVector = set2setdistance(self.Graph, self.Parts[i], self.Parts[j])
        dClasses = sorted(uniquelist(dVector))
        return [[self.Parts[i][idx] for idx in range(len(dVector)) if dVector[idx] == d] for d in dClasses]

    def singlerefinement(self):
        """Return a single refinement of an ordered partition which is not equitable or return the input partition if it
        is equitable """

        nShards = len(self.Parts)
        for i in range(nShards):  # loop until a refinement is found or the partition is found to be equitable
            for j in range(nShards):

                isShattered = self.shattercheck(i, j)  # check if Vi is shattered by Vj
                if isShattered:
                    # print(partition[i])
                    # print('shattered by')
                    # print(partition[j])

                    return self.Parts[:i] + self.shatter(i, j) + self.Parts[i + 1:]  # return the refined partition
        return self.Parts  # the given partition is equitable

    def refine(self):
        """Return an equitable refinement for the partition"""

        isEquitable = False  # initialize flag to determine when P is equitable
        while not isEquitable:
            refineOnce = self.singlerefinement()
            if refineOnce == self.Parts:  # this refinement is equitable
                isEquitable = True
            else:
                print(refineOnce)
                self.Parts = refineOnce  # continue refining
        return

    def isatomic(self):
        """Return true if a partition is atomic i.e. all parts are singletons"""
        subsetLen = [len(S) for S in self.Parts]  # get the vector of part length
        return all([subsetLen[i] == 1 for i in range(len(subsetLen))])

    def nextsplitting(self):
        """Returns the first nontrivial part with respect to the ordering of the partition"""
        nontrivialIdx = [j for j in range(len(self.Parts)) if len(self.Parts[j]) > 1]  # indices for nontrivial parts
        return nontrivialIdx[0]

    def split(self, vertex, idx):
        """Return the splitting of a partition by distinguishing a specified vertex from a given part index"""
        if vertex not in self.Parts[idx]:
            print('specified vertex is not in the specified part')
            raise ValueError

        splitPart = [[vertex], [v for v in self.Parts[idx] if v != vertex]]  # split Vi into [vertex | Vi \setminus {vertex}]
        splitPartition = self.Parts[:idx] + splitPart + self.Parts[idx+1:]

        # A splitting is the equitable refinement of new ordered partition with one vertex distinguished
        splitting = Partition(splitPartition, self.Graph, self.PaintedVertices + [vertex])
        splitting.refine()
        return splitting

    def permutation(self):
        """Return the permutation identified with an atomic partition"""
        if self.isatomic():
            perm = []
            for part in self.Parts:
                perm += part
            return perm
        else:
            print('This partition is not atomic')

    def applyautomorphism(self):
        """Return the networkx graph after applying the automorphism defined by an atomic ordered partition"""
        if self.isatomic():
            perm = self.permutation()
            automorphism = {perm[j]:1+j for j in range(len(perm))}  # automorphism as a dictionary mapping
            imageGraph = nx.relabel_nodes(self.Graph, automorphism)
            return imageGraph
        else:
            print('This partition is not atomic')




# Example partitions from the paper
# W1 = [1]
# W2 = [3, 7, 9]
# W3 = [2, 4, 6, 8]
# W4 = [5]
# P0 = [W1, W2, W3, W4]

P0 = Partition([nodeList], G)
P0.refine()


# splitIdx = P.nextsplitting()
# u = P.Parts[splitIdx][0]  # paint the first vertex of the first nontrivial part
# P1 = P.split(u, splitIdx)
# print(P.isatomic())
# print(P1.Parts)
# print(P1.isatomic())

def branch(partition, vertex, partidx):
    """Append a branch attached to a partition to an existing partition tree"""
    subTree = Tree()
    leaf = partition.split(vertex, partidx)
    subTree.create_node(leaf.PaintedVertices, leaf)
    if leaf.isatomic():
        return subTree

    # recurse onto children nodes to build partition tree depth first
    for v in leaf.Parts[leaf.nextsplitting()]:
        subTree.paste(leaf, branch(leaf, v, leaf.nextsplitting()))

    return subTree

from treelib import Node, Tree
tree = Tree()
tree.create_node(P0.PaintedVertices, P0)  # root node

if not P0.isatomic():
    for v in P0.Parts[P0.nextsplitting()]:
        tree.paste(P0, branch(P0, v, P0.nextsplitting()))

tree.show()
for node in tree.leaves():
    # print(node.identifier.permutation())
    P = node.identifier
    sG = P.applyautomorphism()
    print(lexifyedges(sG))


# P1 = tree.leaves()[0].identifier
# p = P1.permutation()
# G1 = P1.applyautomorphism()
