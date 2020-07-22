import networkx as nx
import numpy as np

# Build an example DSGRN network
interactionList = [[1, -2, 0, 1, 0, 0, 0, 0, 0],
                   [1, 0, -1, 0, -1, 0, 0, 0, 0],
                   [0, 1, -1, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, -1, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, -1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, -1, -1, 0],
                   [0, 0, 0, 0, -1, 0, 1, 0, -1],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0]
                   ]
A = np.array(interactionList)
n = A.shape[0]
nodeList = list(range(n))


# Test shattering functions
def uniquelist(list, idfunc=lambda x: x):
    """uniquify a list of hashable objects"""

    seen = {}
    result = []
    for item in list:
        marker = idfunc(item)
        if marker not in seen:
            seen[marker] = 1
            result.append(item)
    return result


def vertexsetdistance(adjMatrix, vertex, part):
    """Define the distance between a vertex and a part of a partition"""
    d1 = sum(adjMatrix[vertex, part] == 1)  # positive incoming edges to vertex
    d2 = sum(adjMatrix[vertex, part] == -1)  # negative incoming edges to vertex
    d3 = sum(adjMatrix[part, vertex] == 1)  # positive outgoing edges from vertex
    d4 = sum(adjMatrix[part, vertex] == -1)  # negative outgoing edges from vertex
    return tuple([d1, d2, d3, d4])  # return vector valued degree function


def set2setdistance(adjMatrix, part1, part2):
    """Define the vector valued distance between two parts in a partition"""
    return [vertexsetdistance(adjMatrix, v, part2) for v in part1]


def lexifyedges(graph):
    """Return edge set of a networkx graph with lexicographic ordering enforced"""
    # edges = list(graph.edges())
    lexify = lambda edge: (min(edge), max(edge))  # lexify a single edge
    lexEdges = sorted([lexify(edge) for edge in graph.edges()])
    return lexEdges


v = 1
S = [2, 3]
T = [0, 3]
# print(vertexsetdistance(A, v, S))
# print(vertexsetdistance(A, v, T))
a = set2setdistance(A, S, T)


# Define partition class
class Partition:
    def __init__(self, partlist, adjmatrix, paintedvertices=[]):
        self.AdjMatrix = adjmatrix
        self.Parts = partlist
        self.PaintedVertices = paintedvertices

        vertexLabels = []
        for part in self.Parts:
            vertexLabels += part
        self.Vertex = sorted(vertexLabels)


    # def setvertexlabels(self):
    #     """Extract the integer vertex labels from the input partition"""
    #     vLabels = []
    #     for part in self.Parts:
    #         vLabels += part
    #     return sorted(vLabels)

    def shattercheck(self, i, j):
        """Returns true if Vj shatters Vi"""
        return len(uniquelist([vertexsetdistance(self.AdjMatrix, v, self.Parts[j]) for v in self.Parts[i]])) != 1

    def shatter(self, i, j):
        """Return the shattering of Vi by Vj"""
        dVector = set2setdistance(self.AdjMatrix, self.Parts[i], self.Parts[j])
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

        splitPart = [[vertex],
                     [v for v in self.Parts[idx] if v != vertex]]  # split Vi into [vertex | Vi \setminus {vertex}]
        splitPartition = self.Parts[:idx] + splitPart + self.Parts[idx + 1:]

        # A splitting is the equitable refinement of new ordered partition with one vertex distinguished
        splitting = Partition(splitPartition, self.AdjMatrix, self.PaintedVertices + [vertex])
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
            # automorphism = [(perm[j], self.Vertex[j]) for j in range(len(perm))]  # automorphism as a list if (value, image) pairs
            # permMatrix = np.zeros(self.AdjMatrix.shape, dtype=int)  # initialize permutation matrix
            imageMatrix = self.AdjMatrix  # initialize image of adjacency matrix via this automorphism

            imageMatrix = imageMatrix[:, perm] # swap columns
            imageMatrix = imageMatrix[perm, :] # swap rows

            return imageMatrix  # return image of automorphism on adjacency matrix
            # for pair in automorphism:
            #     permMatrix[pair] = 1
            #
            # print(permMatrix.dtype)
            # imageMatrix = permMatrix.transpose()*self.AdjMatrix*permMatrix  # conjugation by a permutation is equivalent to P^tAP
            # print(imageMatrix.dtype)
            # return imageMatrix.astype(int)  # return image of automorphism on adjacency matrix
        else:
            print('This partition is not atomic')


P0 = Partition([nodeList], A)
P0.refine()

N1 = np.array([[1, 1], [1, 0]])
Q1 = Partition([[0, 1]], N1)
Q1.refine()
a = Q1.applyautomorphism()

N2 = np.array([[0, 1], [1, 1]])
Q2 = Partition([[0, 1]], N2)
Q2.refine()
b = Q2.applyautomorphism()

# print(a)
# print(b)

# perm = sorted([(j,j+1) for j in range(0,8,2)] + [(j,j-1) for j in range(1,9,2)])
# pMatrix = np.zeros([9,9])
# for image in perm:
#     pMatrix[image] = 1
# print(pMatrix)
# print(Q2.Parts)

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

# from treelib import Node, Tree
# tree = Tree()
# tree.create_node(P0.PaintedVertices, P0)  # root node
#
# if not P0.isatomic():
#     for v in P0.Parts[P0.nextsplitting()]:
#         tree.paste(P0, branch(P0, v, P0.nextsplitting()))
#
# tree.show()
# for node in tree.leaves():
#     # print(node.identifier.permutation())
#     P = node.identifier
#     sG = P.applyautomorphism()
#     print(lexifyedges(sG))

# P1 = tree.leaves()[0].identifier
# p = P1.permutation()
# G1 = P1.applyautomorphism()
