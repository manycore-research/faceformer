from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer, discretize_edge
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.TopTools import TopTools_SequenceOfShape
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance

from dataset.utils.projection_utils import d3_to_d2, project_shapes, discretize_compound, project_points
from dataset.utils.Edge import Edge
from dataset.utils.Face import Face
from surfaceformer.utils import flatten_list

import numpy as np



class TopoMapper:
    def __init__(self, shape, args):
        self.shape = shape
        self.all_edges = None
        self.all_faces = None
        self.args = args
        self.tol = self.args.tol
        
        # add outline to shape
        outline_edges = self._find_outline_edges()
        self.full_topo = self._add_outline_edges(outline_edges)
        
        # construct all edge-face mappings
        self._construct_mapping()
        
        # project to 2D; each edge has dedge now
        self._project(args.discretize_last)
        
        # remove sewn edges
        sewn_edge_keys = self._find_sewn_edges()
        self._remove_sewn_edges(sewn_edge_keys)

        
    def _find_outline_edges(self):
        hlr_shapes = project_shapes(self.shape, self.args)
        outline_compound = hlr_shapes.OutLineVCompound3d()
        if outline_compound:
            return list(TopologyExplorer(outline_compound).edges())
        return []

    def _num_edges(self, splitshape):
        probing_shape = splitshape.Shape()
        split = BRepFeat_SplitShape(probing_shape)
        return split, len(list(TopologyExplorer(probing_shape).edges()))

    def _add_edge(self, split, edge, num_edge):
        toptool_seq_shape = TopTools_SequenceOfShape()
        toptool_seq_shape.Append(edge)
        add_success = split.Add(toptool_seq_shape)
        split, curr_num_edge = self._num_edges(split)
        add_success = add_success and (curr_num_edge > num_edge)
        return split, curr_num_edge, add_success
        
    def _add_outline_edges(self, outline_edges):
        if len(outline_edges) == 0:
            return TopologyExplorer(self.shape)
        split_edge_num = 0
        while True:
            # repeated split edge until number of edges converge
            split = BRepFeat_SplitShape(self.shape)
            split, num_edge = self._num_edges(split)
            for edge in outline_edges: 
                probing_shape = split.Shape()
                backup_split, split = BRepFeat_SplitShape(probing_shape), BRepFeat_SplitShape(probing_shape)
                split, curr_num_edge, add_success = self._add_edge(split, edge, num_edge)
                if not add_success:
                    # Increase outline tolerance when add fails
                    # fixed tolerance, may need update
                    tol = ShapeFix_ShapeTolerance()
                    tol.SetTolerance(edge, 1)
                    split, curr_num_edge, add_success = self._add_edge(backup_split, edge, num_edge)
                    if not add_success:
                        raise Exception("Fail to add splitting outline")
            if split_edge_num == curr_num_edge:
                break
            split_edge_num = curr_num_edge

        split_shape = split.Shape()
        return TopologyExplorer(split_shape)
        
    def _construct_mapping(self):
        '''
        Construct edge-to-face mapping from wireframe.
        '''
        all_edges = {}
        all_faces = {}

        for face in self.full_topo.faces():
            new_face = Face(face, self)
            all_faces[hash(face)] = new_face

            sharp_edges_wires = list(self.full_topo.wires_from_face(face))
            sharp_edges_3d = []
            for wire in sharp_edges_wires:
                sharp_edges_3d += list(WireExplorer(wire).ordered_edges())

            for edge in sharp_edges_3d:
                edge_id = hash(edge) # same edge has same hash
                
                # create edge
                if edge_id in all_edges:
                    new_edge = all_edges[edge_id]
                    new_edge.add_face(new_face, edge.Orientation())
                else:
                    new_edge = Edge(edge, faces=[new_face], orientations=[edge.Orientation()])
                    all_edges[edge_id] = new_edge
                
                # add edge to face
                new_face.add_edge(new_edge, edge.Orientation())
                
        self.all_faces = all_faces
        self.all_edges = all_edges
        
    def _find_sewn_edges(self):
        '''
        Any edge that occur in any face twice is sewn edge.
        '''
        all_sewn_edge_keys = []
        topo = TopologyExplorer(self.shape)
        for face in topo.faces():
            edge_keys = []
            
            sharp_edges_wires = list(topo.wires_from_face(face))
            sharp_edges_3d = []
            for wire in sharp_edges_wires:
                sharp_edges_3d += list(WireExplorer(wire).ordered_edges())

            for edge in sharp_edges_3d:
                edge_id = hash(edge) # same edge has same hash
                
                # if edge is used twice in a face, it's a sewn edge
                if edge_id in edge_keys:
                    all_sewn_edge_keys.append(edge_id)
                else:
                    edge_keys.append(edge_id)
                    
        return all_sewn_edge_keys
    
    def _remove_sewn_edges(self, sewn_edge_keys):
        '''
        Remove all sewn edge and combine faces.
        '''
        candidate_edges = set()
        for key in sewn_edge_keys:
            # if key in self.all_edges:
            sewn_edge = self.all_edges[key]
            # else:
            #     # sewn edge not found after adding outline

            faces = sewn_edge.faces
            # roll edge sequence
            for face in faces:
                ind = face.keys.index(key)
                face.roll(ind)
            result_face = faces[0]
            for face in faces[1:]:
                pairs = result_face.merge(face)
                if pairs:
                    for pair in pairs:
                        candidate_edges.add(tuple(sorted(pair)))
            
        # merge candidate edges
        for key1, key2 in candidate_edges:
            # check if there's a 4th edge connected to this vertex
            d1, d2 = np.array(self.all_edges[key1].dedge), np.array(self.all_edges[key2].dedge)
            dist = lambda t: np.sum((t[0]-t[1])**2)
            p1, p2 = min([(d1[0], d2[0]), (d1[-1], d2[0]), (d1[0], d2[-1]), (d1[-1], d2[-1])], key=dist)
            vertex = (p1+p2) / 2

            skip = False
            for key in self.all_edges:
                if key == key1 or key == key2 or key in sewn_edge_keys:
                    continue
                e = self.all_edges[key]
                if dist((vertex, e.dedge[0])) < self.tol or dist((vertex, e.dedge[-1])) < self.tol:
                    skip = True
                    break
            
            if not skip:
                self.all_edges[key1].merge(self.all_edges[key2], self)
            
            
        
    def _project(self, discretize_last=False):
        '''
        Project all edges of the shape
        '''
        for edge in list(self.all_edges.values()):
            if not discretize_last:
                sharp_dedge = discretize_edge(edge.edge, self.args.tol)
                edge.dedge3d = project_points(sharp_dedge, self.args)
                edge.dedge = d3_to_d2(edge.dedge3d)
                continue
            sharp_edges_compound = project_shapes(edge.edge, self.args).VCompound()
            if sharp_edges_compound is None:
                # invalid edge - delete
                key = hash(edge.edge)
                del self.all_edges[key]
                # del in face
                for face in edge.faces:
                    face.remove_edge(key)
                continue
            
            dedge = discretize_compound(sharp_edges_compound, self.tol)[0]
            edge.dedge = dedge
    
    # Given a list of edges (they are from the same edge but broken into pieces)
    # project them and return one unified edge
    def _raw_project(self, edges, discretize_last=False):
        if not discretize_last:
            full_2d_dedge = []
            for edge in edges:
                dedge = discretize_edge(edge, self.args.tol)
                full_2d_dedge += d3_to_d2(project_points(dedge, self.args))
            return full_2d_dedge
        sharp_edges_compound = project_shapes(edges, self.args).VCompound()
        dedge = flatten_list(discretize_compound(sharp_edges_compound, self.tol)[:len(edges)])
        return dedge
    
    # return x,y,z directions in camera world
    def get_dominant_directions(self):
        # project origin, x, y, z
        points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
        origin, x, y, z = project_points(points, self.args)
        origin, x, y, z = [np.array(p) for p in [origin, x, y, z]]
        return (x - origin).tolist(), (y - origin).tolist(), (z - origin).tolist()