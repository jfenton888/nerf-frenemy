# Nerfrenemy Install 

Both this repository and Nerfstudio should be cloned into a common parent repository, called something like `nerfrenemy_ws` 

Starting from [Nerfstudio Installation](https://docs.nerf.studio/quickstart/installation.html)
```
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

pip uninstall torch torchvision functorch tinycudann
```

Then for using Cuda 11.7
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```

```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```


Then clone Nerfstudio to the workspace directory. Currently this would actually require cloning my fork which handles adding overriding the pixel sampling approach for some dictionary keys, like those coming from posecnn.

I am not installing in editable mode because Nerfrenemy then has a hard time finding the source files. If you need to update then just re-run the install script
```
git clone git@github.com:jfenton888/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install .
```

```
ns-install-cli
```

Nerfacto should be completely installed now

For installing Nerfrenemy, return to the workspace directory

```
git clone git@github.com:jfenton888/nerfrenemy.git
cd nerfrenemy
pip install -e .
```

```
ns-install-cli
```




## File Structure
We recommend the following file structure:

```
├── frenemy
│   ├── __init__.py
│   ├── frenemy_config.py
│   ├── frenemy_model.py
│   ├── task_models
│   │   ├── base_task.py
│   │   ├── posecnn_task.py
│   ├── PoseCNN
│   │   ├── posecnn.py
│   │   ├── posecnn_model.pth
│   ├── posecnn_dataset.py
│   ├── posecnn_segmentation_dataset.py
│   ├── propspose_datamanger.py
│   ├── propspose_segmentation_datamanger.py
├── utils
│   ├── datset_modifier.py
│   ├── metric_over_ablation.py
│   ├── ...
├── pyproject.toml
```


## Running the new method
First you will need to train a nerfacto model on the dataset that you wish to modify. Keep in mind that the directory from which you run this command matters as it determines where the model gets saved. I chose to train most model from inside the nerfstudio directory and stored datasets outside in the workspace directory
```
ns-train nerfacto --data ../datasets/PROPS-NeRF
```


```
ns-train method-template --data ../datasets/PROPS-NeRF --pipeline.model.nerf-path outputs/PROPS-NeRF/nerfacto/TIME/nerfstudio_models/
```

The flag that are important to have at te correct paths

```
--data
--pipeline.model.nerf-path
--pipeline.model.task-model.model-path
--pipeline.model.task-model.dataset-dir
```