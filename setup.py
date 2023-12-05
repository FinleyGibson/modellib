from setuptools import setup, find_packages

setup(
    name='modellib',
    version='0.1.0',
    author='Finley Gibson',
    author_email='f.j.gibson@exeter.ac.uk',
    packages=find_packages(include=['modellib', 'modellib.*']),
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'scikit-learn',
        'parameterized',
        'xgboost',
    ],
    extras_require={
        'interactive': ['matplotlib', 'jupyter'],
        'book': ['jupyter', 'jupyter-book'],
    }
)
