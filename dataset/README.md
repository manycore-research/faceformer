## Our Dataset

### Annotation format

We save the annotation in a JSON format, including 2D edge points and face loops by edge indices.

```json
{
    'edges': [
        [...],  # edge 1
        [...],  # edge 2
        ...
    ],
    'faces_indices': [
        [...],  # face 1
        [...],  # face 2
        ...
    ],
}
```

## Prepare Dataset

Here, we provide tools to filter and parse data in [ABC dataset](https://archive.nyu.edu/handle/2451/43778). Please download `step`, `stat`, `obj`, and `feat`.

## Reorganize ABC dataset directory

Remove the middle level folder after unzipping ABC dataset.

```bash
python reorganize_dataset_dirs.py --root $ABC_ROOT_DIR
```

```bash
# Original ABC Dataset Structure
root
└── step
    └──00000050
       └── 00000050.step
# Reorganized ABC Dataset structure
root
└── step
    └── 00000050.step
```

### Dataset Directory Structure

```
root
├── step
│   └── 00000050.step
├── json
│   └── 00000050.json
├── face_png
│   └── 00000050_{face_index}.png
├── face_svg
│   └── 00000050_{face_index}.svg
├── png
│   └── 00000050.png
└── svg
    └── 00000050.svg
```

## Command Lines

#### Data Generation

In each model's [config](configs), we detail the specific options needed to generate dataset of the correct format.

```bash
# parse the entire ABC dataset
python dataset/prepare_data.py --root $ABC_ROOT_DIR --id_list dataset/dataset_gen_logs/filtered_id_list.json > dataset/dataset_gen_logs/error.txt
# parse a specific object (for debugging a single data)
python dataset/prepare_data.py --root $ABC_ROOT_DIR --name $8_DIGIT_ID
```

#### Dataset Filtering

Filter ABC objects by similarity
```bash
# 1. filter by topology similarity
python dataset/filters/filter_topology.py --root $ABC_ROOT_DIR
# 2. render the three views of the entire ABC dataset
python dataset/filters/3view_render.py --root $ABC_ROOT_DIR --id_list dataset/dataset_gen_logs/filtered_id_list.json > dataset/dataset_gen_logs/3view_error.txt
# 3. filter by three-view similarity
python dataset/filters/filter_3view.py --root $ABC_ROOT_DIR
```

Filter ABC objects by thickness
```bash
python dataset/filters/filter_thickness.py --root $ABC_ROOT_DIR --save_root $DIR_FOR_TEMP_DATA
```

Filter ABC objects by complexity
```bash
# By default, $MAX_FACE_SEQ = 128, $MAX_NUM_EDGE = 64
python dataset/filters/filter_length.py --root $ABC_ROOT_DIR --face_seq_max $MAX_FACE_SEQ --num_edge_max $MAX_NUM_EDGE
```

Filter Generated Co-edge Data by Face Encloseness
```bash
# Assume prepare_data.py has finished and all json files have generated
python dataset/tests/check_faces_enclosed.py --root $ABC_ROOT_DIR --tol 1e-4 --remove
# Regenerate train/valid/test splits
python dataset/prepare_data.py --root $ABC_ROOT_DIR --only_split
```