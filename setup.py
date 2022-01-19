from setuptools import setup

setup(
    name='cluster_tfidf',
    version='1.0',
    description='A module to compute the Cluster TFIDF method',
    author='VFMR',
    packages=['cluster_tfidf'],  #same as name
    install_requires=['pandas', 'scikit-learn', 'numpy', 'tqdm'], #external packages as dependencies
)