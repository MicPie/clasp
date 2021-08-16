from setuptools import setup, find_packages

setup(
  name = 'bioseq-clasp',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'CLASP - CLIP for biosequences and their annotation data',
  author = 'MicPie',
  author_email = '',
  url = 'https://github.com/MicPie/clasp',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'contrastive learning',
    'proteomics'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6',
    'ftfy',
    'regex',
    'requests',
    'matplotlib'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
