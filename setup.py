from setuptools import find_packages,setup #package that provides tools for packaging python projects
from typing import List

HYPEN_E_DOT='-e.'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    return requirements

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup( # constains metadeta or basic information

    name='Fault detection',
    version='0.0.1',
    author='ishika kaur bhurji',
    author_email='ishikakaur745@gmail.com',
    install_requirements=('requirements.txt'),
    packages=find_packages()
)
