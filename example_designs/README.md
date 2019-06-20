# Examples

This folder contains a sequence of runner scripts that allow you to initialize
a training procedure for each of the supported environments mentioned in the
[README](https://github.com/AboudyKreidieh/Hierarchical-Actor-Critc-HAC-#environments)
within the repository's main directory.

TODO: describing the benchmarking and sample commands sections for each
environment.

Before continuing to the examples, we recommend **installing h-baselines** by 
executing the following 
[installation instructions](https://github.com/AboudyKreidieh/Hierarchical-Actor-Critc-HAC-#installation). 
Moreover, if you have decided to create a conda environment, be sure to 
activate the environment before running any of the examples. You can do so by 
running the following command in terminal:

```bash
source activate h-baselines
```

## Contents

* [Pendulum](#pendulum)
  * [Benchmarking (Pendulum)](#benchmarking-pendulum)
  * [Sample Commands (Pendulum)](#sample-commands-pendulum)
* [UR5](#ur5)
  * [Benchmarking (UR5)](#benchmarking-ur5)
  * [Sample Commands (UR5)](#sample-commands-ur5)
* [Creating a Custom Example](#creating-a-custom-example)

## Pendulum

### Benchmarking (Pendulum)

TODO: add plots of benchmarking results

### Sample Commands (Pendulum)

* To run the HAC algorithm with one layer, type:
  
  ```bash
  python train.py "pendulum" --layers=1 --time_scale=1000
  ```

* To run the HAC algorithm with two layers, type:

  ```bash
  python train.py "pendulum" --layers=2 --time_scale=32
  ```

* To run the HAC algorithm with three layers, type:

  ```bash
  python train.py "pendulum" --layers=3 --time_scale=10
  ```

## UR5

### Benchmarking (UR5)

TODO: add plots of benchmarking results

### Sample Commands (UR5)

* To run the HAC algorithm with one layer, type:

  ```bash
  python train.py "ur5" --layers=1 --time_scale=600
  ```

* To run the HAC algorithm with two layers, type:

  ```bash
  python train.py "ur5" --layers=2 --time_scale=25
  ```

* To run the HAC algorithm with three layers, type:

  ```bash
  python train.py "ur5" --layers=3 --time_scale=10
  ```

## Creating a Custom Example

TODO: talk about the format of these runner scripts, and use that as a
replacement for design_agent_and_env.py in hac/.
