import numpy as np

from dataset.tests.check_faces_enclosed import is_face_enclosed
from faceformer.utils import flatten_list


# For each face, if it is enclosed, sort its loops
def filter_faces_by_encloseness(edges, faces, tol):
    # find corresponding edges from face indices
    filtered_faces = []
    for face_type, face in faces:
        all_face_loops = is_face_enclosed(edges, face, tol)
        if all_face_loops:
            # roll enclosed loops so smallest index is at the front
            all_face_loops = [tuple(np.roll(loop, -np.argmin(loop), axis=0).astype(int).tolist()) for loop in all_face_loops]
            # loops are ordered by first index
            all_face_loops = sorted(all_face_loops, key=lambda x: x[0])
            filtered_faces.append((face_type, tuple(all_face_loops)))

    return filtered_faces

# Two coedges that represent the same edge should not be used in the same face
def filter_faces_by_coedge(pairings, faces):
    filtered_faces = []
    used_indices = set()
    for face in faces:
        indices = flatten_list(face[1])
        drop_face = False
        for index in indices:
            if index in pairings:
                index = pairings[index]
                if index in used_indices:
                    drop_face = True
                    break
            used_indices.add(index)
        if not drop_face:
            filtered_faces.append(face)

    return filtered_faces

def map_coedge_into_edges(pairings, indices):
    new_indices = []
    for i in indices:
        if str(i) in pairings:
            new_indices.append(pairings[str(i)])
        else:
            new_indices.append(i)
    return new_indices
