import json
import os

import numpy as np
import torch

from faceformer.utils import flatten_list


# TODO: try using bilinear interpolation for all cases
def sample_points(edge, num_samples=50):
    if len(edge) == 2:
        return sample_points_on_line(edge, num_samples)
    return sample_points_on_curve(edge, num_samples)


def sample_points_on_line(line, num_samples):
    t = np.linspace(0, 1, num_samples)
    x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    x = x1 + (x2-x1) * t
    y = y1 + (y2-y1) * t
    return np.vstack([x, y]).T


def sample_points_on_curve(curve, num_samples):
    samples = np.linspace(0, len(curve)-1, num_samples).round(0).astype(int)
    curve = np.array(curve)
    return curve[samples]


class ABCDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, datafile_path, config):
        super(ABCDataset, self).__init__()
        self.root_dir = root_dir
        self.info_files = self.parse_splits_list(datafile_path)

        # input shape L x P x D
        self.num_points_per_line = config.num_points_per_line  # P
        self.num_lines = config.num_lines  # L
        self.point_dim = config.point_dim  # D
        # output shape S
        self.label_seq_length = config.label_seq_length

        self.token = config.token

        # preload all files
        self.raw_datas = []
        for info_file in self.info_files:
            with open(os.path.join(self.root_dir, info_file), "r") as f:
                self.raw_datas.append(json.loads(f.read()))
        
        
    def __len__(self):
        return len(self.info_files)

    def __getitem__(self, index):
        raw_data = self.raw_datas[index]

        edges, faces_indices = raw_data['edges'], raw_data['faces_indices']

        input = np.zeros(
            (self.num_lines, self.num_points_per_line, self.point_dim), dtype=np.float32)
        for i, edge in enumerate(edges):
            input[i, :self.num_points_per_line] = sample_points(
                edge, self.num_points_per_line)

        input_mask = np.ones(self.num_lines, dtype=np.bool)
        input_mask[:len(edges)] = 0

        label = np.ones(self.label_seq_length, dtype=np.int) * self.token.PAD
        label[0] = self.token.SOS
        curr_pos = 0
        for face in faces_indices:
            if not isinstance(face[0], int):
                face = flatten_list(face)
            curr_pos += 1
            label[curr_pos:curr_pos+len(face)] = face
            # shift face indices for special tokens
            label[curr_pos:curr_pos+len(face)] += self.token.len
            curr_pos += len(face)
            label[curr_pos] = self.token.SEP
        label[curr_pos] = self.token.EOS
        label_mask = (label == self.token.PAD)

        data = {
            'id': index,
            'input': input,
            'label': label,
            'num_input': len(edges),
            'num_label': curr_pos+1,
            'input_mask': input_mask,
            'label_mask': label_mask,
            'name': self.info_files[index]
        }

        return data

    def parse_splits_list(self, splits):
        """ Returns a list of info_file paths
        Args:
            splits (list of strings): each item is a path to a .json data file 
                or a path to a .txt file containing a list of .json's relative paths from root.
        """
        if isinstance(splits, str):
            splits = splits.split()
        info_files = []
        for split in splits:
            ext = os.path.splitext(split)[1]
            split_path = os.path.join(self.root_dir, split)
            # split_path = os.path.join("/root/ablation/polys-test", split)
            if ext == '.json':
                info_files.append(split_path)
            elif ext == '.txt':
                info_files += [info_file.rstrip() for info_file in open(split_path, 'r')]
            else:
                raise NotImplementedError('%s not a valid info_file type' % split)
        return info_files
