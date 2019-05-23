from os.path import dirname, realpath
from setuptools import find_packages, setup
from hac.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='hac',
    version=__version__,
    packages=find_packages(),
    description="A repository of high-performing hierarchical reinforcement "
                "learning models and algorithms.",
    long_description=open("README.md").read(),
    url="https://github.com/AboudyKreidieh/hac",
    keywords="hierarchical-reinforcement-learning deep-learning python",
    install_requires=_read_requirements_file(),
    zip_safe=False,
)
