import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name='bpe_surgery',
      version='0.0.1',
      url='',
      discription="Morphology Aware Tokenization",
      long_description=readme,
      long_description_content_type='text/markdown',
      author='Zaid Alyafeai',
      author_email='alyafey22@gmail.com',
      license='MIT',
      packages=['bpe_surgery'],
      install_requires=required,
      python_requires=">=3.6",
      include_package_data=True,
      zip_safe=False,
      )
