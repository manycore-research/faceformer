# faceformer

<p align="center">
  <img src="assets/teaser.gif">
</p>

## Paper

**Neural Face Identification in a 2D Wireframe Projection of a Manifold Object**

[Kehan Wang](https://jason-khan.github.io/),
[Jia Zheng](https://bertjiazheng.github.io/),
[Zihan Zhou](https://zihan-z.github.io/)

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022

[[arXiv](https://arxiv.org/abs/2203.04229)] [[Supp. Material](https://drive.google.com/file/d/1dy_4-H86NGziDqbYIYIrFnZDRBY0gu5T/view)]

## Requirements

```bash
conda env create --file environment.yml
conda activate faceformer
```

## Download Dataset

We use CAD mechanical models from [ABC dataset](https://archive.nyu.edu/handle/2451/43778). In order to reproduce our results, we also release the dataset used in the paper [here](https://drive.google.com/drive/u/2/folders/1ynMD02E5FWlCPmQkWyjHdq4Zhe8DIXE2). If you would like to build the dataset by yourself, please refer to [here](dataset/README.md).

## Evaluation

### Face Identification Model
Trained models can be downloaded [here](https://drive.google.com/drive/u/2/folders/1oEoN_GzS36obLjvOlwFrOpWo0N7oh-fS).
```bash
python main.py --config-file configs/{MODEL_NAME}.yml --test_ckpt trained_models/{MODEL_NAME}.ckpt
```

Face predictions will be saved to `lightning_logs/version_{LATEST}/json`.

### 3D Reconstruction

```bash
# wireframe reconstruction
python reconstruction/reconstruct_to_wireframe.py --root lightning_logs/version_{LATEST}
# surface reconstruction
python reconstruction/reconstruct_to_mesh.py --root lightning_logs/version_{LATEST}
```

Reconstructed wireframes (*.ply*) or meshes (*obj*) files will be saved to `lightning_logs/version_{LATEST}/{ply/obj}`

## Train a Model from Scratch

```bash
python main.py --config_file configs/{MODEL_NAME}.yml
```

## Citation

Please cite `faceformer` in your publications if it helps your research:

```bibtex
@inproceedings{faceformer,
  title     = {Neural Face Identification in a 2D Wireframe Projection of a Manifold Object},
  author    = {Kehan Wang and Jia Zheng and Zihan Zhou},
  booktitle = {Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

## License

[MIT license](LICENSE)