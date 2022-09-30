import sys
import os
from setuptools import setup, find_packages

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RL_Toy"))
from RL_Toy.version import VERSION

setup(
    name="RL_Toy",
    version=VERSION,
    description="Simple examples",
    url="https://github.com/Bobobert/RL_Toy",
    author="Roberto-Esteban Lopez",
    author_email="robertolopez94@outlook.com",
    license="MIT",
    install_requires=["matplotlib", "gif", "gym"],
    python_requires=">=3.6",
    packages=[package for package in find_packages()
                if package.startswith('RL_Toy')],
)

