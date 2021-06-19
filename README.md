## Implicit field learning for unsupervised anomaly detection in medical images

## Citing 

If you use this implementation, please cite [paper](https://arxiv.org/abs/2106.05214):
```
@article{marimont2021implicit,
  title={Implicit field learning for unsupervised anomaly detection in medical images},
  author={Naval Marimont, Sergio and Tarroni, Giacomo},
  journal={arXiv preprint arXiv:2106.05214},
  year={2021}
}
```

## Instructions

#### Image preprocessing
Given that no augmentations were used, we extracted and stored the images already pre-processed. Pre-processing includes padding, clip intensities, normalize, downsample (optional), discretize using pixel intensity clusters, mode-pooling filter. Script provided below does this for HCP and BraTS'18 datasets:

```
python extract_and_preproc_brains.py --dataset <HCP or BraTS> --source <root directory of Brats18 / HCP with the original nii.gz> --target <directory to store pre-processed dataset> --res <160 or full> --cluster_centers <pix_cluster_centers_brain.cp (file with the k-Means cluster centers)> --mode_pool <2 or 3 or not specified if not used>
```

Pre-processed files are pickle files with `img_extended = namedtuple( 'img_extended', ('img', 'seg', 'cid') )`, containing the pre-processed Image, Segmentation (`None` in HCP dataset) and the identifier of the case (original file name).


#### Train a Model

The network architecture and implementation is based on DeepSDF [repository](https://github.com/facebookresearch/DeepSDF/). To train an experiment run the following:

```
python train_network.py --experiment <experiment directory> --continue <latest or epoch number>
```

##### Experiment Layout

Each experiment is organized in an "experiment directory", which collects all of the data relevant to a particular experiment. The structure is as follows:

```
<experiment_name>/
    specs.json
    Logs.pth
    LatentCodes/
        <Epoch>.pth
    ModelParameters/
        <Epoch>.pth
    OptimizerParameters/
        <Epoch>.pth
```

The only file that is required to begin an experiment is 'specs.json', which sets the parameters, network architecture, and data to be used for the experiment. Examples of spec.json are provided.

Note that the first time that a data loader is instantiated on the training set, a file called "dict_cid.k" is created in the training set directory. This file contains a dictionary assigning a unique sequential Integer to each case id.

#### Evaluate

Evaluation scripts are implemented in [Implicit Fields Evaluation jupyter notebook][6]


## Team

Sergio Naval Marimont, Giacomo Tarroni


## License

Implicit field learning for unsupervised anomaly detection in medical images is relased under the MIT License. We use a substantial portion of DeepSDF which is relased under the MIT License. See the [LICENSE file][5] for more details.

[5]: https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE
[6]: https://github.com/snavalm/ifl_unsup_anom_det/blob/main/Implicit%20Fields%20Evaluation.ipynb

