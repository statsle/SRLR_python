from setuptools import setup, find_packages

# Package metadata
NAME = 'SRLR'
VERSION = '0.1.8'
DESCRIPTION = 'A package for sketched ridgeless estimator simulations, optimizing generalization. Identify the best sketching size to minimize out-of-sample risks. Stable risk curves in optimally sketched estimator eliminate peaks found in full-sample estimator. SRLR offers practical method to discover the ideal sketching size.'
AUTHOR = 'Siyue Yang'
EMAIL = 'syue.yang@mail.utoronto.ca'
URL = 'https://github.com/statsle/SRLR_python'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.6'

# Package dependencies
INSTALL_REQUIRES = [
    'numpy>=1.21',
    'scipy>=1.7',
    'joblib>=1.3',
    'scikit-learn>=1.0',
    'matplotlib>=3.2',
    'scienceplots>=2.1',
]

# Long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages("src"),
    package_dir={"": "src"},
    keywords=['python', 'sketched ridgeless linear regression'],
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux"
    ],
    setup_requires=['wheel']
)
