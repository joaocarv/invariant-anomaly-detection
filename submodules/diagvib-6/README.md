# DiagViB-6

Common deep neural networks (DNNs) for image classification have been shown to rely on shortcut opportunities (SO) in the form of predictive and easy-to-represent visual factors. This is known as shortcut learning and leads to impaired generalization. In this work, we show that common DNNs also suffer from shortcut learning when predicting only basic visual object factors of variation (FoV) such as shape, color, or texture. Here, we introduce the Diagnostic Vision Benchmark suite *DiagViB-6*, which includes datasets and metrics to study a network’s shortcut vulnerability and generalization capability for six independent FoV. In particular, *DiagViB-6* allows controlling the type and degree of SO and GO in a dataset.

For more information about this work, please read our [ICCV 2021 paper](https://arxiv.org/abs/2108.05779):

> Eulig, E., Saranrittichai, P., Mummadi, C., Rambach, K., Beluch, W., Shi, X., & Fischer, V. (2021). DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

## Table of Contents
- [Installation](#installation)
- [Dataset generation](#dataset-generation)
  * [Generate study specifications](#generate-study-specifications)
  * [Preprocess MNIST](#preprocess-mnist)
- [Benchmark a model](#benchmark-a-model)
  * [Implement a new model](#implement-a-new-model)
  * [Start single training run](#start-single-training-run)
  * [Run training on a study](#run-training-on-a-study)
  * [Plot results](#plot-results)
- [Questions and Reference](#questions-and-reference)


## Installation

First we recommend to setup a python environment using the provided `environment.yml` and install the DiagViB-6 benchmark:

```
conda env create -f environment.yml
source activate diagvibsix
pip install -e .
```

## Dataset generation

### Generate study specifications

Calling the following three scripts will generate abstract dataset specification for all studies and store them in `tmp/studies/`

```
python diagvibsix/generate_study/generate_studies_ZSO_ZGO_FGO.py
python diagvibsix/generate_study/generate_studies_CGO-123.py
python diagvibsix/generate_study/generate_studies_CHGO.py
```

This will generate all studies (ZSO, ZGO, CHGO, FGO, CGO) for all different factor combinations and the studies can be inspected in `./tmp/diagvibsix/studies/`.
Generally study naming follows the pattern `CORR-F1-F2_PRED-F3`, where `F1` and
`F2` name the two correlated factors in the dataset and `F3` names the predicted factor.


### Preprocess MNIST

As a next step we will preprocess the MNIST dataset to slightly bigger digits for better texture application:

* Download the MNIST dataset in npz-format, e.g., from [Keras](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz).
* Adapt the folder name `SHARED_LOADPATH_MNIST` in `diagvibsix/dataset/config.py` accordingly.
* Pre-process MNIST: This will load, pre-process and store the resulting MNIST dataset in `SHARED_SAVEPATH_MNIST`.

```
python diagvibsix/dataset/preprocess_mnist.py
```

## Benchmark a model

### Implement a new model
To evaluate a new model on our benchmark you need to implement its architecture in `./models` and create a trainer in 
`./trainer/trainer_setup`. Depending on your method you may also need to redefine the class methods `train_step`, 
`val_step`, and `test_step` of the `BaseTrainer` (`./trainer/base_trainer.py`).

### Start single training run

A sample model (ResNet18) is provided in `./models/resnet.py` and its trainer in `./trainer/trainer_setup`. It can be trained and 
evaluated using `python run.py --method ResNet18Trainer`.
```
usage: run.py [-h] [-dev DEVICE] [--results_path RESULTS_PATH] [--study_folder STUDY_FOLDER] [--study STUDY] [--experiment EXPERIMENT] [--dataset_sample DATASET_SAMPLE]
              [--dataset_seed DATASET_SEED] [--cache] [--method METHOD] [--class_criterion CLASS_CRITERION] [--training_seed TRAINING_SEED] [--num_epochs NUM_EPOCHS]
              [--mbs MBS] [--num_workers NUM_WORKERS] [--optimizer OPTIMIZER] [--lr LR] [--adam_b1 ADAM_B1] [--adam_b2 ADAM_B2] [--sgd_momentum SGD_MOMENTUM]
              [--sgd_dampening SGD_DAMPENING]

optional arguments:
  -h, --help            show this help message and exit
  -dev DEVICE, --device DEVICE
                        Cuda device to use. If -1, cpu is used instead.
  --results_path RESULTS_PATH
                        The general save path.
  --study_folder STUDY_FOLDER
                        Folder the study specification were generated in.
  --study STUDY         Name of the study.
  --experiment EXPERIMENT
                        Name of the experiment.
  --dataset_sample DATASET_SAMPLE
                        The sample of the dataset.
  --dataset_seed DATASET_SEED
                        Seed to use for dataset setup.
  --cache               Cache the dataset in a .pkl file.
  --method METHOD       Method to use. This should be one of the classes in trainer/.
  --class_criterion CLASS_CRITERION
                        Criterion used for classification.
  --training_seed TRAINING_SEED
                        Seed with which torch is initialized.
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for.
  --mbs MBS             Minibatch size to use.
  --num_workers NUM_WORKERS
                        Number of workers to use for loading and pre-processing.
  --optimizer OPTIMIZER
                        Optimizer to use. Must be one of sgd | adam | rmsprop
  --lr LR               Learning rate.
  --adam_b1 ADAM_B1     b1 to use for adam
  --adam_b2 ADAM_B2     b2 to use for adam
  --sgd_momentum SGD_MOMENTUM
                        Momentum factor for SGD training
  --sgd_dampening SGD_DAMPENING
                        Dampening for momentum factor for SGD training
```

### Run training on a study
To run a model on a study it needs to be run on all dataset samples of all
experiments of that study. This can be done using `run_study.py`. The model and
all non-dataset related hyperparamters are provided through YAML-files.  

```
python run_study.py --study study_ZGO --hp trainer/resnet18config.yml
```
runs the ResNet18 on the entire ZGO study. We can also use this script to run
only a single experiment or a single dataset sample of an experiment. E.g.

```
python run_study.py --study study_ZGO/CORR-position-hue_PRED-position/0 --hp trainer/resnet18config.yml
```
runs the ResNet18 on the first dataset sample of the ZGO task where hue and position are 
correlated and the position is predicted.
> :exclamation: **Note**
Depending on the study and your model, the benchmark may take very long if only
run on a single machine. We recommend running *DiagViB-6* on a cluster. For this,
you'll want to change l. 36 of `run_study.py` to a call to a submit script for your cluster.

### Plot results
Once a model is run on an entire study, we can evaluate its performance easily by 
plotting the results similar to how we did in the paper. For example, after running the 
ResNet18 on the entire ZGO study (using the command above) we can call 
```
python plot_results.py --study study_ZGO --hp trainer/resnet18config.yml
```
Similar plots can be generated for all other studies by changing the `--study` argument accordingly.

## Questions and Reference
Please contact [Elias Eulig](mailto:elias@eeulig.com?subject=[GitHub]%20DiagViB-6)
or [Volker Fischer](mailto:volker.fischer@de.bosch.com?subject=[GitHub]%20DiagViB-6) with
any questions about our work and reference it, if it benefits your research:
```
@InProceedings{Eulig_2021_ICCV,
author = {Eulig, Elias and Saranrittichai, Piyapat and Mummadi, Chaithanya Kumar and Rambach, Kilian and Beluch, William and Shi, Xiahan and Fischer, Volker},
title = {DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```
