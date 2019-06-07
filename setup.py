import os
from os.path import dirname, realpath
from setuptools import find_packages, setup
from hac.version import __version__


# This is used to install MuJoCo only when outside the travis build.
test_flag = os.environ.get('TEST_FLAG', "False")
if test_flag == "False":
    extras = {'all': ['mujoco-py<2.1,>=2.0']}
else:
    extras = {'all': []}


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='h-baselines',
    version=__version__,
    packages=find_packages(),
    description="A repository of high-performing hierarchical reinforcement "
                "learning models and algorithms.",
    long_description=open("README.md").read(),
    url="https://github.com/AboudyKreidieh/h-baselines",
    keywords="hierarchical-reinforcement-learning deep-learning python",
    install_requires=_read_requirements_file(),
    extras_require=extras,
)
