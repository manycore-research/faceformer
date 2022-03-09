'''
Dataset Generation Integrity Test
- Check if each face is enclosed
'''

import json, os, argparse
from functools import partial
from tqdm.contrib.concurrent import process_map

# check if e1's end meets e2's start
def e1_connects_e2(e1, e2, tol):
    return abs(e1[-1][0] - e2[0][0]) < tol and \
            abs(e1[-1][1] - e2[0][1]) < tol

# face can be composed of multiple enclosed loops
# check if all oriented-edges form loops
# return all loops in list of lists if face is enclosed
def is_face_enclosed(edges, face_indices, tol):
    all_loops = []
    curr_loop = []
    to_close = None # the start of an enclosed cycle, to be closed
    last_edge = None
    for ind in face_indices:
        if isinstance(ind, tuple):
            i, o = ind
            edge = edges[i][::-1] if o else edges[i]
        else:
            if ind < len(edges):
                edge = edges[ind]
            else:
                continue
        if to_close is None:
            to_close = edge
        else:
            # make sure the current edge connects to the last edge
            if not e1_connects_e2(last_edge, edge, tol):
                return False
                
        last_edge = edge
        curr_loop.append(ind)
        if e1_connects_e2(edge, to_close, tol):
            # close the current cycle
            to_close = None 
            all_loops.append(curr_loop)
            curr_loop = []
    return all_loops if to_close is None else False

def check_enclosed(name, args):
    path = os.path.join(args.root, 'json', f'{name}.json')
    with open(path, 'r') as f:
        data = json.load(f)
        edges = data['edges']
        faces_indices = data['faces_indices']

    for face_indices in faces_indices:
        if not is_face_enclosed(edges, face_indices, args.tol):
            if args.remove:
                # remove json from dataset
                os.remove(path)
            print(f"{name} contains unclosed face")
            return

def main(args):
    names = []
    for name in sorted(os.listdir(os.path.join(args.root, 'json'))):
        names.append(name[:8])
    
    process_map(
        partial(check_enclosed, args=args), names,
        max_workers=args.num_cores, chunksize=args.num_chunks
    )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data",
                        help='dataset root.')
    parser.add_argument('--name', type=str, default=None,
                        help='filename.')
    # default to 3e-4 since the discretization tolerance is 1e-4
    parser.add_argument('--tol', type=float,
                        default=3e-4, help='same point tolerance.')
    parser.add_argument('--num_cores', type=int,
                        default=40, help='number of processors.')
    parser.add_argument('--num_chunks', type=int,
                        default=10, help='number of chunk.')
    parser.add_argument('--remove', action='store_true')

    args = parser.parse_args()

    if args.name is None:
        main(args)
    else:
        check_enclosed(args.name, args)