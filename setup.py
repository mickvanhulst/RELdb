from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='REL_database',
    version='0.0.1',

    description='Pretrained word embeddings in Python.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/mickvanhulst/REL_database',

    # Author details
    author='Johannes Michael van Hulst',
    author_email='mick.vanhulst@gmail.com',

    # Choose your license
    license='MIT',

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['tqdm', 'requests', 'numpy', 'gensim'],
)
