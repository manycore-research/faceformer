<div align="center">

# Neural Face Identification in a 2D Wireframe Projection of a Manifold Object

<h4>
  <a href='https://jason-khan.github.io/' target='_blank'>Kehan Wang</a>
  ·
  <a href='https://bertjiazheng.github.io/' target='_blank'>Jia Zheng</a>
  ·
  <a href='https://zihan-z.github.io/' target='_blank'>Zihan Zhou</a>
</h4>

<h4>
  IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2022
</h4>

[![arXiv](http://img.shields.io/badge/arXiv-2203.04229-B31B1B.svg)](https://arxiv.org/abs/2203.04229)
[![Conference](https://img.shields.io/badge/CVPR-2022-4b44ce.svg)](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Neural_Face_Identification_in_a_2D_Wireframe_Projection_of_a_CVPR_2022_paper.html)

<img src="assets/teaser.gif">

</div>

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

## Acknowledgement

The work was done during Kehan Wang's internship at Manycore Tech Inc.

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
