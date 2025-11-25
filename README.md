# The F1TENTH Gym environment

This is the repository for a residual policy learner on the F1Tenth Gym environment.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
We recommend installing the simulation inside a virtualenv. The project was tested using a conda environment.
You can find [documentation](https://docs.conda.io/en/latest/) for conda here. 

```bash
conda create --name rl_model python=3.8
conda activate rl_model
git clone https://github.com/ajm1312/f1tenth_gym_rl.git
cd f1tenth_gym_rl
pip install -e .
```

Next, the global paths must be updated in the config file

First, find the global path by typing
```bash
pwd
```
then copying the resulting path. Next, navigate to the config file.

```bash
cd gym/f110_gym/envs/
code .
```

Navigate to the config file and change the following paths:
```bash
map_path: '<path>/maps/GLC_pit_rbring1'
model_path: '<path>/gym/f110_gym/envs/models'
wpt_path: '<path>/maps/GLC_pit_rbring1_extended_raceline.csv'
model_path: '<path>/gym/f110_gym/envs/models/'
```

To run a simple test using a configured model, run the test command
```bash
python3 test.py
```


## Known issues
- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.20
```
And you might see an error similar to
```
f110-gym 0.2.1 requires pyglet<1.5, but you have pyglet 1.5.20 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
