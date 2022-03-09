import json
import os

import numpy as np
import torch


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


class ABCDataset_Parallel(torch.utils.data.Dataset):

    def __init__(self, root_dir, datafile_path, config):
        super(ABCDataset_Parallel, self).__init__()

        self.root_dir = root_dir
        self.info_files = self.parse_splits_list(datafile_path)

        # input shape L x P x D
        self.num_points_per_line = config.num_points_per_line  # P
        self.num_lines = config.num_lines  # L
        self.point_dim = config.point_dim  # D

        # output shape F x T
        self.max_num_faces = config.max_num_faces # F
        self.max_face_length = config.max_face_length # T

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

        input_mask = np.ones(self.num_lines, dtype=np.bool) # L
        input_mask[:len(edges)] = 0

        # F x T
        label = np.ones((self.num_lines, self.max_face_length), dtype=np.int) * self.token.PAD
        ind = 0
        # each face: [(loop 1), ..., (loop n)]
        for face_with_type in faces_indices:
            type, face = face_with_type
            # only allow Plane - 0, Cylinder - 1, Other - 2
            if type > 1:
                type = 2
            # type offset set to 1
            type += self.token.face_type_offset
            # each loop rolls itself
            for loop in face:
                for i in range(len(loop)):
                    # construct new seq
                    rotated_loop = np.roll(loop, i, axis=0).tolist()
                    new_seq = rotated_loop 
                    for other_loop in face:
                        if other_loop != loop:
                            new_seq += other_loop
                    label[ind, :len(new_seq)] = new_seq
                    # shift face indices for special tokens
                    label[ind, :len(new_seq)] += self.token.len
                    label[ind, len(new_seq)] = type
                    ind += 1
        for i in range(ind, self.num_lines):
            label[i, 0] = self.token.len - 1 # set to Other type of face
        label_mask = (label == self.token.PAD)

        data = {
            'id': index,
            'input': input,
            'label': label,
            'num_input': len(edges),
            'num_faces': len(faces_indices),
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
