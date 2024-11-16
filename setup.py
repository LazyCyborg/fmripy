from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fmripy",  # Changed to match your package name
    version="0.1.0",
    author="Alexander Engelmark",  # Updated author name
    author_email="alexander.engelmark@gmail.com",
    description="A package for fMRI connectivity analysis including searchlight, ROI-ROI, and seed-based approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LazyCyborg/fmripy",
    packages=find_packages(exclude=['notebooks*']),  # Added exclude for notebooks
    install_requires=[
        'numpy',
        'pandas',
        'nibabel',
        'nilearn',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'matplotlib',
        'seaborn',
        'torch'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    # Added package_data to include potential non-Python files
    package_data={
        'fmripy': ['*.json', '*.txt']
    },
    # Added include_package_data to ensure MANIFEST.in is used
    include_package_data=True,
    # Added zip_safe flag
    zip_safe=False
)