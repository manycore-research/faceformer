import argparse
import json
import os
from functools import partial
from surfaceformer.utils import flatten_list

import numpy as np
import svgwrite
from cairosvg import svg2png
from matplotlib.cm import get_cmap as colormap
from tqdm.contrib.concurrent import process_map


def discretized_edge_to_svg_polyline(points):
    """ Returns a svgwrite.Path for the edge, and the 2d bounding box
    """
    return svgwrite.shapes.Polyline(points, fill="none", class_='vectorEffectClass')

def save_svg_groups(groups_of_edges, filename, args):
    discretized_edges = flatten_list(groups_of_edges)
    all_edges = flatten_list(discretized_edges)

    # compute bounding box
    min_x, min_y = np.min(all_edges, axis=0) - args.png_padding
    max_x, max_y = np.max(all_edges, axis=0) + args.png_padding
    width, height = max_x - min_x, max_y - min_y


    # build the svg drawing
    dwg = svgwrite.Drawing(filename, (args.width, args.height), debug=True)
    dwg.viewbox(min_x, min_y, width, height)

    # make sure line width stays constant
    # https://github.com/mozman/svgwrite/issues/38
    dwg.defs.add(
        dwg.style(".vectorEffectClass {\nvector-effect: non-scaling-stroke;\n}"))

    n = len(groups_of_edges) + 1
    cmap = (colormap('coolwarm')(np.linspace(0, 1, n))[:, :3]*255).astype(np.uint8)
    np.random.seed(args.seed)
    cmap = cmap[np.random.permutation(n), :]
    for index, group in enumerate(groups_of_edges):
        color = ",".join([str(c) for c in cmap[index]])
        for edge in group:
            polyline = discretized_edge_to_svg_polyline(edge)
            polyline.stroke(f"rgb({color})",
                            width=args.line_width, linecap="round")
            dwg.add(polyline)
    # export to string or file according to the user choice
    dwg.save()


def save_svg(discretized_edges, filename, args, color='black'):
    # compute polylines for all edges
    polylines = [discretized_edge_to_svg_polyline(
        edge) for edge in discretized_edges]

    all_edges = flatten_list(discretized_edges)

    # compute bounding box
    min_x, min_y = np.min(all_edges, axis=0) - args.png_padding
    max_x, max_y = np.max(all_edges, axis=0) + args.png_padding
    width, height = max_x - min_x, max_y - min_y


    # build the svg drawing
    dwg = svgwrite.Drawing(filename, (args.width, args.height), debug=True)
    dwg.viewbox(min_x, min_y, width, height)

    # make sure line width stays constant
    # https://github.com/mozman/svgwrite/issues/38
    dwg.defs.add(
        dwg.style(".vectorEffectClass {\nvector-effect: non-scaling-stroke;\n}"))

    n = len(polylines) + 1
    cmap = (colormap('jet')(np.linspace(0, 1, n))[:, :3]*255).astype(np.uint8)
    cmap = cmap[np.random.permutation(n), :]

    for index, (dedge, polyline) in enumerate(zip(discretized_edges, polylines)):
        if color != 'black':
            color = ",".join([str(c) for c in cmap[index]])
            color = f"rgb({color})"
        polyline.stroke(color,
                        width=args.line_width, linecap="round")
        dwg.add(polyline)
        # add a circle at the beginning of the edge
        dwg.add(dwg.circle(dedge[0], r=4/256, fill='black'))
        
    # export to string or file according to the user choice
    dwg.save()


def save_png(name, args, prefix=''):
    svg2png(
        bytestring=open(
            os.path.join(args.root, prefix+'svg', f'{name}.svg'), 'rb').read(),
        output_width=args.width,
        output_height=args.height,
        background_color='white',
        write_to=os.path.join(args.root, prefix+'png', f'{name}.png')
    )
        

def json_to_svg_png(name, args):
    json_filename = os.path.join(args.root, 'json', f'{name}.json')
    with open(json_filename, "r") as f:
        data = json.loads(f.read())
        edges, faces_indices = data['edges'], data['faces_indices']
        # reconstruct faces
        for index, face_indices in enumerate(faces_indices):
            face = [edges[ind] for ind in face_indices]
            filename = os.path.join(
                args.root, args.prefix+'face_svg', f'{name}_{index}.svg')
            save_svg(face, filename, args)

        save_svg(edges, os.path.join(
            args.root, args.prefix+'svg', f'{name}.svg'), args)
        # generate_pngs(name, args, prefix=args.prefix)


def main(args):
    os.makedirs(os.path.join(args.root, args.prefix+'svg'), exist_ok=True)
    os.makedirs(os.path.join(args.root, args.prefix+'face_svg'), exist_ok=True)
    os.makedirs(os.path.join(args.root, args.prefix+'png'), exist_ok=True)
    os.makedirs(os.path.join(args.root, args.prefix+'face_png'), exist_ok=True)

    names = []
    for name in sorted(os.listdir(os.path.join(args.root, 'step'))):
        if not name.endswith('.step'):
            continue

        names.append(os.path.splitext(name)[0])

    process_map(
        partial(json_to_svg_png, args=args), names,
        max_workers=args.num_cores, chunksize=args.num_chunks
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data",
                        help='dataset root.')
    parser.add_argument('--num_cores', type=int,
                        default=1, help='number of processors.')
    parser.add_argument('--num_chunks', type=int,
                        default=16, help='number of chunk.')
    parser.add_argument('--line_width', type=str,
                        default=str(3/256), help='svg line width.')
    parser.add_argument('--name', type=str, default=None,
                        help='filename.')
    parser.add_argument('--width', type=int,
                        default=256, help='svg width.')
    parser.add_argument('--height', type=int,
                        default=256, help='svg height.')
    parser.add_argument('--prefix', type=str, default='json_',
                        help='filename prefix for generated svg and png')
    args = parser.parse_args()

    if args.name is None:
        main(args)
    else:
        json_to_svg_png(args.name, args)
