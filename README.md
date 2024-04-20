# nerfstudio-method-template
Template repository for creating and registering methods in Nerfstudio.

## File Structure
We recommend the following file structure:

```
├── my_method
│   ├── __init__.py
│   ├── my_config.py
│   ├── custom_pipeline.py [optional]
│   ├── custom_model.py [optional]
│   ├── custom_field.py [optional]
│   ├── custom_datamanger.py [optional]
│   ├── custom_dataparser.py [optional]
│   ├── ...
├── pyproject.toml
```

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "method-template". To train with it, run the command:
```
ns-train method-template --data [PATH]
```




## Helpful Commands from this project

Set environment variables that get added or removed on activation of the conda environment
`$CONDA_PREFIX/etc/conda/activate.d` and `$CONDA_PREFIX/etc/conda/deactivate.d`

#### These lines are to be added
From https://stackoverflow.com/questions/68221962/nvcc-not-found-but-cuda-runs-fine

export CUDA_HOME=$CONDA_PREFIX
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH




### Issues 

For some reason currently getting this when activating the environment when nerfstudio is pip installed (goes away when uninstalled)
```
hotel@hotel-sim:~/nerfrenemy_ws/nerfstudio$ conda activate nerfstudio
optimizer: command not found
inside: command not found
is: No such file or directory
inside: command not found
Warning:: command not found
Unable: command not found
libio_e57.so:: command not found
bash: /home/hotel/anaconda3/envs/nerfstudio/lib/python3.8/site-packages/nerfstudio/scripts/completions/bash/_ns-export: line 6: syntax error near unexpected token `('
bash: /home/hotel/anaconda3/envs/nerfstudio/lib/python3.8/site-packages/nerfstudio/scripts/completions/bash/_ns-export: line 6: `Cannot load library /home/hotel/anaconda3/envs/nerfstudio/lib/python3.8/site-packages/pymeshlab/lib/plugins/libio_e57.so: (/lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0)'
optimizer: command not found
inside: command not found
is: No such file or directory
inside: command not found
optimizer: command not found
inside: command not found
is: No such file or directory
inside: command not found
optimizer: command not found
inside: command not found
is: No such file or directory
inside: command not found
```