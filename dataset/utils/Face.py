from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

class Face:
    '''
    Face is a collection of non-repeating edges
    '''
    def __init__(self, face, topo):
        surface = BRepAdaptor_Surface(face)
        self.face = face
        self.face_type = surface.GetType()
        self.topo = topo
        self.edges = []
        self.edge_orientations = []
        self.keys = []

        # get face parametric values
        if self.face_type == GeomAbs_Plane:
            plane = surface.Surface().Plane()
            # plane parameters: Location, XAxis, YAxis, ZAxis, Coefficients
            self.parameters = {'Location': self._get_vector_parameters(plane.Location()), 
                               'XAxis': self._get_axis_parameters(plane.XAxis()),
                               'YAxis': self._get_axis_parameters(plane.YAxis()), 
                               'Normal': self._get_axis_parameters(plane.Axis()), 
                               'Coefficients': plane.Coefficients()}
        elif self.face_type == GeomAbs_Cylinder:
            cylinder = surface.Surface().Cylinder()
            # cylinder parameters: Location, XAxis, YAxis, ZAxis, Coefficients, Radius
            self.parameters = {'Location': self._get_vector_parameters(cylinder.Location()),
                               'XAxis': self._get_axis_parameters(cylinder.XAxis()),
                               'YAxis': self._get_axis_parameters(cylinder.YAxis()),
                               'Normal': self._get_axis_parameters(cylinder.Axis()), 
                               'Coefficients': cylinder.Coefficients(),
                               'Radius': cylinder.Radius()}
        else:
            self.parameters = None
    
    # Given an OCC vector
    # Return XYZ
    def _get_vector_parameters(self, vector):
        return vector.X(), vector.Y(), vector.Z()

    # Given an axis
    # Return Location(XYZ), Direction(XYZ)
    def _get_axis_parameters(self, axis):
        location = self._get_vector_parameters(axis.Location())
        direction = self._get_vector_parameters(axis.Direction())
        return location, direction

    def add_edge(self, edge, orientation):
        self.edges.append(edge)
        self.edge_orientations.append(orientation)
        self.keys.append(hash(edge))
        
    def remove_edge(self, key):
        ind = self.keys.index(key)
        del self.keys[ind]
        del self.edges[ind]
        del self.edge_orientations[ind]
        
    def get_oriented_dedges(self, is_3d=False):
        return [e.get_oriented_dedge(o, is_3d) for e, o in zip(self.edges, self.edge_orientations)]

    def get_edge_ind_and_orientation(self):
        return [(e.index, o) for e, o in zip(self.edges, self.edge_orientations)]
    
    def roll(self, n):
        self.edges = np.roll(self.edges, -n, axis=0).tolist()
        self.edge_orientations = np.roll(self.edge_orientations, -n, axis=0).tolist()
        self.keys = np.roll(self.keys, -n, axis=0).tolist()
        
    def merge(self, other):
        '''
        Merge faces on sewn edge. 
        Assume both faces are rolled properly with sewn edge at the front.
        edge[0] is sewn edge.
        Return edge merging candidates
        '''
        assert isinstance(other, Face), 'Cannot merge face with non-face'
        sewn_edge = self.edges[0]
        if self == other:
            self.edges = self.edges[1:]
            self.edge_orientations = self.edge_orientations[1:]
            self.keys = self.keys[1:]
            key = hash(sewn_edge)
            if key in self.keys:
                self.remove_edge(key)
            
            del self.topo.all_edges[hash(sewn_edge)]
            return None
        
        # change faces in edge
        for edge in other.edges[1:]:
            i = edge.faces.index(other)
            edge.faces[i] = self
        
        # candidate merge edges
        candidates = [(self.keys[1], other.keys[-1]), (self.keys[-1], other.keys[1])]

        # merge face
        self.edges = self.edges[1:] + other.edges[1:]
        self.edge_orientations = self.edge_orientations[1:] + other.edge_orientations[1:]
        self.keys = self.keys[1:] + other.keys[1:]
        if self.face_type != other.face_type:
            self.face_type = 10 # set face type to other
        

        # remove sewn edge and other face in topo
        del self.topo.all_edges[hash(sewn_edge)]
        del self.topo.all_faces[hash(other.face)]

        return candidates