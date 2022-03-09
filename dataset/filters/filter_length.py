import argparse
import json
import os

from tqdm import tqdm


def main(args):
    if args.clean_start:
        with open("dataset/dataset_gen_logs/filtered_id_list.json", 'r') as f:
            names = json.load(f)
    else:
        names = [os.path.splitext(name)[0] for name in os.listdir(os.path.join(args.root, 'json'))]

    filtered_names = []

    for name in tqdm(names):
        path = os.path.join(args.root, 'json', f'{name}.json')
        with open(path, 'r') as f:
            data = json.load(f)
        total_len = 0
        for face in data["faces_indices"]:
            total_len += 1+len(face)
        total_len += 1
        if total_len < args.face_seq_max and len(data['edges']) < args.num_edge_max:
            filtered_names.append(name)

    with open("dataset/dataset_gen_logs/filtered_id_list.json", 'w') as f:
        json.dump(filtered_names, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/tianhan/data',
                        help='dataset root')
    parser.add_argument('--face_seq_max', type=int, default=128,
                        help='max length for the constructed face label')
    parser.add_argument('--num_edge_max', type=int, default=64,
                        help='max number of edges in a shape')
    parser.add_argument('--clean_start', action='store_true',
                        help='start from a clean id list')
    args = parser.parse_args()

    main(args)
