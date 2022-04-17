# STARK-LT
The STARK tracker for the VOT2021-LT challenge


## Install the environment

**Option1**: Use the Anaconda

```yml
conda create -n stark python=3.6
conda activate stark
bash install.sh
pip install lap
pip install cython_bbox
pip install shapely
pip install sklearn
pip install mmcv==0.4.0
pip install terminaltables
cd ./global_track/_submodules/mmdetection
(if there exists 'build' package, you need to delete it and re-build.)
pip install -r requirements/build.txt
python setup.py develop
```

重点重点！安装 `vot-toolkit-python`

```shell
pip install git+https://github.com/votchallenge/vot-toolkit-python
```


## Set paths
Run the following command to set paths for this project

```shell
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

After running this command, you can also modify paths by editing these two files

```shell
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Download the model checkpoints

Download [STARKST_ep0050.pth.tar](https://drive.google.com/file/d/1OKazQszrFs-xxrlptXDkCOs33N4vGhXm/view?usp=sharing) and put it under checkpoints/train/stark_st2/baseline  
Download [STARKST_ep0500.pth.tar](https://drive.google.com/file/d/1cfSAQYJMi7gn_ZCQZMhVp3pfqsSVTBEK/view?usp=sharing) and put it under checkpoints/train/stark_ref/baseline


## Test and evaluate STARK on benchmarks

**VOT2021-LT**

- Modify the <PATH_OF_STARK> in [trackers.ini](VOT21/LT/stark_st50_ref_baseline_R0/trackers.ini) to the absolute path of the STARK project on your local machine.
- Modify the base_path in `vot_path.py` to the absolute path of the STARK project on your local machine.
- VOT2021-LT dataset should be put in under `stark_st50_ref_baseline_R0` dir, called `sequences`.

```shell
cd VOT21/LT/stark_st50_ref_baseline_R0
bash exp.sh
```

# 问题

## `trax` 编译

在 `trax` 目录下执行：

```shell
mkdir build
cd build
cmake ..
make
```

> 参考链接：`trax` 官网 https://trax.readthedocs.io/en/latest/tutorial_compiling.html


## 运行

```bash
bash exp.sh
```

- `vot connt found`: 没有安装 `vot-toolkit-python`，解决方法：

```shell
pip install git+https://github.com/votchallenge/vot-toolkit-python
```

> 参考链接1：https://www.votchallenge.net/howto/tutorial_python.html
> 参考链接2：https://github.com/votchallenge/toolkit/issues/21


# STARK 执行命令

## Train STARK
Training with multiple GPUs using DDP
```shell
# STARK-S50
python tracking/train.py --script stark_s --config baseline --save_dir . --mode multiple --nproc_per_node 8  # STARK-S50
# STARK-ST50
python tracking/train.py --script stark_st1 --config baseline --save_dir . --mode multiple --nproc_per_node 8  # STARK-ST50 Stage1
python tracking/train.py --script stark_st2 --config baseline --save_dir . --mode multiple --nproc_per_node 8 --script_prv stark_st1 --config_prv baseline  # STARK-ST50 Stage2
# STARK-ST101
python tracking/train.py --script stark_st1 --config baseline_R101 --save_dir . --mode multiple --nproc_per_node 8  # STARK-ST101 Stage1
python tracking/train.py --script stark_st2 --config baseline_R101 --save_dir . --mode multiple --nproc_per_node 8 --script_prv stark_st1 --config_prv baseline_R101  # STARK-ST101 Stage2
```

(Optionally) Debugging training with a single GPU
```shell
python tracking/train.py --script stark_s --config baseline --save_dir . --mode single
```
## Test and evaluate STARK on benchmarks

- LaSOT
```shell
python tracking/test.py stark_st baseline --dataset lasot --threads 16
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```shell
python tracking/test.py stark_st baseline_got10k_only --dataset got10k_test --threads 16
python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_only
```
- TrackingNet
```shell
python tracking/test.py stark_st baseline --dataset trackingnet --threads 16
python lib/test/utils/transform_trackingnet.py --tracker_name stark_st --cfg_name baseline
```
- VOT2020  
Before evaluating "STARK+AR" on VOT2020, please install some extra packages following [external/AR/README.md](external/AR/README.md)
```shell
cd external/vot20/<workspace_dir>
export PYTHONPATH=<path to the stark project>:$PYTHONPATH
bash exp.sh
```
- VOT2020-LT
```shell
cd external/vot20_lt/<workspace_dir>
export PYTHONPATH=<path to the stark project>:$PYTHONPATH
bash exp.sh
```
## Test FLOPs, Params, and Speed
```shell
# Profiling STARK-S50 model
python tracking/profile_model.py --script stark_s --config baseline
# Profiling STARK-ST50 model
python tracking/profile_model.py --script stark_st2 --config baseline
# Profiling STARK-ST101 model
python tracking/profile_model.py --script stark_st2 --config baseline_R101
# Profiling STARK-Lightning-X-trt
python tracking/profile_model_lightning_X_trt.py
```

## Model Zoo
The trained models, the training logs, and the raw tracking results are provided in the [model zoo](MODEL_ZOO.md)
