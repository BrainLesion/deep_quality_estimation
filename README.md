# Deep Quality Estimation

[![Python Versions](https://img.shields.io/pypi/pyversions/deep_quality_estimation)](https://pypi.org/project/deep_quality_estimation/)
[![Stable Version](https://img.shields.io/pypi/v/deep_quality_estimation?label=stable)](https://pypi.python.org/pypi/deep_quality_estimation/)
[![Documentation Status](https://readthedocs.org/projects/deep_quality_estimation/badge/?version=latest)](http://deep_quality_estimation.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/deep_quality_estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/deep_quality_estimation/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/deep_quality_estimation/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/deep_quality_estimation) -->

Quality prediction for brain tumor segmentation on scale ranging from 1 to 6 stars &#x2B50;.  
Can be used to estimate the quality of a segmentation for evaluation purposes or as e.g. as part of a loss function during model training.

> [!NOTE]  
> This package expects images in atlas space and segmentation labels in brats style, i.e.
> - `label 1` is the necrotic and non-enhancing tumor core
> - `label 2` is the peritumoral edema
> - `label 3` is the GD-enhancing tumor (used to be label 4 in older datasets, both are supported)

## Installation

With a Python 3.9+ environment, you can install `deep_quality_estimation` directly from [PyPI](https://pypi.org/project/deep_quality_estimation/):

```bash
pip install deep_quality_estimation
```


## Use Cases and Tutorials

A minimal example to predict the quality of a segmentation could look like this:

```python
    from deep_quality_estimation import DQE

    # shown parameters are default values but can be adapted to usecase
    dqe = DQE(device="cuda", cuda_devices="0") 

    # inputs can be Paths (str or pathlib.Path object), NumPy NDArrays or a mix
    mean_score, scores_per_view = dqe.predict(
        t1c="t1c.nii.gz",
        t1="t1.nii.gz",
        t2="t2.nii.gz",
        flair="flair.nii.gz",
        segmentation="segmentation.nii.gz",
    )
```


## Citation

If you use deep_quality_estimation in your research, please cite it to support the development!

https://arxiv.org/abs/2205.10355
```
@misc{kofler2022deepqualityestimationcreating,
      title={Deep Quality Estimation: Creating Surrogate Models for Human Quality Ratings}, 
      author={Florian Kofler and Ivan Ezhov and Lucas Fidon and Izabela Horvath and Ezequiel de la Rosa and John LaMaster and Hongwei Li and Tom Finck and Suprosanna Shit and Johannes Paetzold and Spyridon Bakas and Marie Piraud and Jan Kirschke and Tom Vercauteren and Claus Zimmer and Benedikt Wiestler and Bjoern Menze},
      year={2022},
      eprint={2205.10355},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2205.10355}, 
}
```

## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/deep_quality_estimation/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
