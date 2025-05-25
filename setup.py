from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str):
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name = "VisiTranscrible_AI",
    version = "1.0",
    author = "Arinjoy Nandy",
    author_email = "arinjoynandy2019@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)
