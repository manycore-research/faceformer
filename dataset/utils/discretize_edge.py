from functools import cmp_to_key

import numpy as np


class DiscretizedEdge:
    def __init__(self, points, smaller_edge=None, edge3d=None):
        self.points = points
        self.index = None
        self.smaller_edge = smaller_edge
        self.edge3d = edge3d

    def __eq__(self, obj):
        return isinstance(obj, DiscretizedEdge) and obj.points == self.points

    def correct_edge_direction(self, tolerance=1e-10):
        """
        Given a discretized_edge
        Point edge in the direction of smaller coordinate to larger coordinate.
        """
        if self.is_enclosed(tolerance):
            self.sort_enclosing_edge()
        else:
            if comp_points(self.points[0], self.points[-1]) > 0:
                # reverse edge
                self.points = list(reversed(self.points))

    # check for enclosed polyline with tolerance
    def is_enclosed(self, tolerance):
        return abs(self.points[0][0] - self.points[-1][0]) < tolerance and \
            abs(self.points[0][1] - self.points[-1][1]) < tolerance

    # rotate points for an enclosing edge
    def sort_enclosing_edge(self):
        # take out the repeating start/end
        enclosing_edge = self.points[1:]

        # find smallest starting point
        edge_array = np.array(enclosing_edge)
        d_edge = np.roll(
            edge_array, -np.argmin(edge_array[:, 0]), axis=0).tolist()

        # sort direction clock-wise by y-axis
        if d_edge[1][1] > d_edge[-1][1]:
            d_edge.append(d_edge[0])
        else:
            d_edge = [d_edge[0]] + list(reversed(d_edge))

        self.points = d_edge


# rank coordinates first by x, then by y
def comp_points(p1, p2):
    if p1[0] == p2[0]:
        return p1[1] - p2[1]
    return p1[0] - p2[0]

# rank edges in sequence of the points
# assuming edges themselves are sorted


def comp_edges(e1, e2):
    e1, e2 = e1.points, e2.points
    N = min(len(e1), len(e2))
    for i in range(N):
        diff = comp_points(e1[i], e2[i])
        if diff == 0:
            continue
        return diff
    return 0


def sort_edges_by_coordinate(edges):
    return sorted(edges, key=cmp_to_key(comp_edges))


def comp_face_by_index(f1, f2):
    N = min(len(f1), len(f2))
    for i in range(N):
        diff = f1[i] - f2[i]
        if diff == 0:
            continue
        return diff
    return 0


def sort_faces_by_indices(faces):
    return sorted(faces, key=cmp_to_key(comp_face_by_index))

