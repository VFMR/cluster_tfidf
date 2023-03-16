from setuptools import setup

setup(
    name='cluster_tfidf',
    version='0.1',
    description='A module to compute the Cluster TFIDF method',
    author='VFMR',
    packages=['cluster_tfidf'],  #same as name
    install_requires=['pandas', 
                      'scikit-learn',
                      'numpy',
                      'tqdm'],
    extras_require={
        'dev': [
            'pytest',
            'sphinx',
            'sphinx_rtd_theme'
            ]
        }
)
