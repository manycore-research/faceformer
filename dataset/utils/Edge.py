import numpy as np

class Edge:
    '''
    Edge is unique by the edge's hash
    Each edge should have two faces
    '''
    
    def __init__(self, edge, faces=[], orientations=[], dedge=None, index=None, DiscretizedEdge=None, dedge3d=None):
        self.edge = edge
        self.edges = [edge]
        self.faces = faces
        self.orientations = orientations
        self.dedge = dedge
        self.dedge3d = dedge3d
        self.index = index # index among all edges in TopoMapper, for construct faces
        self.DiscretizedEdge = DiscretizedEdge
        
    def add_face(self, face, orientation):
        self.faces.append(face)
        self.orientations.append(orientation)
        assert len(self.faces) <= 2, "Too many faces for one edge"
        
    def get_oriented_dedge(self, orientation, is_3d=False):
        '''
        Discretized edge is saved as normal orientation.
        return reversed when orientation is reversed.
        '''
        if is_3d:
            return self.dedge3d[::-1] if orientation else self.dedge3d
        return self.dedge[::-1] if orientation else self.dedge
        
    def __hash__(self):
        return hash(self.edge)
    
    def __eq__(self, other):
        return isinstance(other, Edge) and hash(self) == hash(other)
    
    def same_orientation(self, other):
        dist1 = np.sum(abs(np.array(self.dedge[-1]) - np.array(other.dedge[0])))
        dist2 = np.sum(abs(np.array(other.dedge[-1]) - np.array(self.dedge[0])))
        return dist1 < dist2
    
    def merge(self, other, topo):
        '''
        Merge two edges, considering the faces being merged.
        Not assigning self.edge to None so hash of the edge is still available.
        Assuming the orientation of dedge is the same.
        '''
        assert isinstance(other, Edge), 'Cannot merge edge with non-edge'
        # check orientation by looking at start and end of two edges
        if self.same_orientation(other):
            self.dedge = self.dedge + other.dedge
            self.edges = self.edges + other.edges
        else:
            self.dedge = other.dedge + self.dedge
            self.edges = other.edges + self.edges
        
        # remove other in its faces
        for face in other.faces:
            i = face.keys.index(hash(other.edge))
            del face.edges[i]
            del face.edge_orientations[i]
            del face.keys[i]
        
        # remove other edge from topo
        del topo.all_edges[hash(other.edge)]
        return self
