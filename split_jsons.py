import argparse
import os
import numpy as np
import shutil

def prepare_splits(args):
    names = []
    os.makedirs(os.path.join(args.root, 'json'), exist_ok=True)
    for name in sorted(os.listdir(args.root)):
        names.append(name[:8])
        shutil.move(os.path.join(args.root, name), os.path.join(args.root, "json"))

    np.random.seed(args.seed)
    np.random.shuffle(names)
    train_ratio, valid_ratio, test_ratio = args.split
    trainlist, validlist, testlist = np.split(names, [int(
        len(names) * train_ratio), int(len(names) * (train_ratio + valid_ratio))])

    np.savetxt(os.path.join(args.root, 'train.txt'), trainlist, fmt="json/%s.json")
    np.savetxt(os.path.join(args.root, 'valid.txt'), validlist, fmt="json/%s.json")
    np.savetxt(os.path.join(args.root, 'test.txt'), testlist, fmt="json/%s.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./ours",
                        help='dataset root. All files under root are all .json files.')
    parser.add_argument('--seed', type=int, default=42,
                        help='numpy random seed')
    parser.add_argument('--split', nargs="+", type=int, 
                        default=[0.93, 0.02, 0.05],
                        help='train/valid/test split ratio')

    args = parser.parse_args()

    prepare_splits(args)
