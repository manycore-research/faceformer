import numpy as np
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
from OCC.Extend.TopologyUtils import discretize_edge


def construct_connected_cylinder(edges, edge_inds, tol=1e-4):
    '''
    Given lines and their indices,
    Form a loop of edges and return the loop's edges, indices and edge directions(1/-1).
    '''

    # group edges by their intersections
    groups = {}
    edge_ind_to_intersection = {}
    for edge, edge_ind in zip(edges, edge_inds):
        start, end = tuple(edge[0]), tuple(edge[-1])
        start_found, end_found = False, False
        # find start's group
        for intersection in groups:
            if dist(start, intersection) < tol:
                groups[intersection].append((edge, 1, edge_ind))
                start_found = True
                break
        if not start_found:
            groups[start] = [(edge, 1, edge_ind)]
            intersection = start
        
        if edge_ind not in edge_ind_to_intersection:
            edge_ind_to_intersection[edge_ind] = [intersection]
        else:
            edge_ind_to_intersection[edge_ind].append(intersection)

        # find end's group
        for intersection in groups:
            if dist(end, intersection) < tol:
                groups[intersection].append((edge, -1, edge_ind))
                end_found = True
                break
        if not end_found:
            groups[end] = [(edge, -1, edge_ind)]
            intersection = end
        
        if edge_ind not in edge_ind_to_intersection:
            edge_ind_to_intersection[edge_ind] = [intersection]
        else:
            edge_ind_to_intersection[edge_ind].append(intersection)
    # fix one corner to be the origin. Generate a circle from the origin
    for intersection, edge_inter in groups.items():
        assert len(edge_inter) == 2, "more than two edges intersect at one intersection"
        edge1, edge2 = edge_inter[0][0], edge_inter[1][0]
        # intersection of a line and a curve is a real intersection
        if is_straight_line(edge1) or is_straight_line(edge2):
            origin = intersection
            break

    # construct the circle
    circle = []
    circle_inds = []
    dirs = []
    next_point = origin
    count = 0
    while True:
        for edge, direction, edge_ind in groups[next_point]:
            if edge_ind not in circle_inds:
                break
        circle.append(edge[::direction])
        circle_inds.append(edge_ind)
        dirs.append(direction)
        # find the next point
        for intersection in edge_ind_to_intersection[edge_ind]:
            if tuple(next_point) != tuple(intersection):
                next_point = intersection
                break
        if next_point == origin:
            break
        count += 1
        if count >= 10:
            print("cylinder construction failed")
            break

    # return circle indices in sequence
    return circle, circle_inds, dirs
        

def construct_connected_cycle(edges, edge_inds, tol=1e-4):
    '''
    Given lines and their indices,
    Form a loop of edges and return the loop's edges, indices and edge directions(1/-1).
    '''

    # group edges by their intersections
    groups = {}
    edge_ind_to_intersection = {}
    for edge, edge_ind in zip(edges, edge_inds):
        start, end = tuple(edge[0]), tuple(edge[-1])
        start_found, end_found = False, False
        # find start's group
        for intersection in groups:
            if dist(start, intersection) < tol:
                groups[intersection].append((edge, 1, edge_ind))
                start_found = True
                break
        if not start_found:
            groups[start] = [(edge, 1, edge_ind)]
            intersection = start
        
        if edge_ind not in edge_ind_to_intersection:
            edge_ind_to_intersection[edge_ind] = [intersection]
        else:
            edge_ind_to_intersection[edge_ind].append(intersection)

        # find end's group
        for intersection in groups:
            if dist(end, intersection) < tol:
                groups[intersection].append((edge, -1, edge_ind))
                end_found = True
                break
        if not end_found:
            groups[end] = [(edge, -1, edge_ind)]
            intersection = end
        
        if edge_ind not in edge_ind_to_intersection:
            edge_ind_to_intersection[edge_ind] = [intersection]
        else:
            edge_ind_to_intersection[edge_ind].append(intersection)
    

    # construct circles
    all_circles = []
    all_circle_inds = []
    all_dirs = []
    while len(groups) > 0:
        origin = list(groups.keys())[0]
        circle = []
        circle_inds = []
        dirs = []
        next_point = origin
        skip = False
        while True:
            if next_point not in groups:
                skip = True
                break
            for edge, direction, edge_ind in groups[next_point]:
                if edge_ind not in circle_inds:
                    break
            circle.append(edge[::direction])
            circle_inds.append(edge_ind)
            dirs.append(direction)
            del groups[next_point]

            # find the next point
            for intersection in edge_ind_to_intersection[edge_ind]:
                if tuple(next_point) != tuple(intersection):
                    next_point = intersection
                    break
            if next_point == origin:
                break
        if not skip:
            all_circles.append(circle)
            all_circle_inds.append(circle_inds)
            all_dirs.append(dirs)
    # return circle indices in sequence
    return all_circles, all_circle_inds, all_dirs
                


def check_parallel(v1, v2, tol=1e-10):
    return np.abs(np.dot(v1, v2)) > (1 - tol)

def fit_curve(p1, p2, p3):
    '''
    Given three 3D points, fit a circle to the points.
    Return the discretized curve between p1-p3-p2
    '''
    center, radius, normal = find_circle_center(p1, p2, p3)

    # construct opencascade circle
    center = gp_Pnt(center[0], center[1], center[2])
    normal = gp_Vec(normal[0], normal[1], normal[2])
    ax = gp_Ax2(center, gp_Dir(normal))
    circle = gp_Circ(ax, radius)
    circle_edge = BRepBuilderAPI_MakeEdge(circle).Edge()
    pts = discretize_edge(circle_edge, deflection=1e-5)
    return find_curve_between_points(pts, p1, p2, p3)

def find_circle_center(p1, p2, p3):
    # triangle "edges"
    t = np.array(p2 - p1)
    u = np.array(p3 - p1)
    v = np.array(p3 - p2)

    # triangle normal
    w = np.cross(t, u)
    wsl = w.dot(w)

    # helpers
    iwsl2 = 1.0 / (2.0 * wsl)
    tt = t.dot(t)
    uu = u.dot(u)

    # result circle
    center = p1 + (u*tt*(u.dot(v)) - t*uu*(t.dot(v))) * iwsl2
    radius = np.sqrt(tt * uu * (v.dot(v)) * iwsl2 / 2)
    normal   = w / np.sqrt(wsl)
    return center, radius, normal

def find_curve_between_points(pts, p1, p2, p3):
    pts = np.array(pts)
    p1_ind = np.argmin(np.linalg.norm(pts - p1, axis=1))
    p2_ind = np.argmin(np.linalg.norm(pts - p2, axis=1))
    p1_ind, p2_ind = min(p1_ind, p2_ind), max(p1_ind, p2_ind)
    right_direction = p3 - pts[p1_ind]
    v1 = pts[(p1_ind+1) % (len(pts)-1)] - pts[p1_ind]
    # selecting p1_ind to p2_ind if angle is acute
    if np.dot(v1, right_direction) > 0:
        pts = pts[p1_ind:p2_ind+1]
    # selecting everything else if angle is obtuse
    else:
        pts = np.vstack([pts[p2_ind:], pts[:p1_ind+1]])
    return pts

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_straight_line(line):
    return len(line) == 2
