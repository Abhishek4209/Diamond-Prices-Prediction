from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirments(file_path:str)->List[str]:
    requirements = []
    
    with open(file_path, 'r') as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
        return requirements


setup(
    name='DiamondPricesPrediction',
    version='0.0.1',
    author='Abhishek Upadhyay',
    author_email='abhishekupadhyay9336@gmail.com',
    install_requires=get_requirments('requirements.txt'),
    packages=find_packages()
    
)