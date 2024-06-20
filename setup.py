from setuptools import setup, find_packages

setup(
    name='nwdaf_anomaly_detection',  # Replace with your package's name
    version='0.1.0',         # The initial release version
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of what your package does',
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        # Dependencies listed in your requirements.txt can be added here
        # e.g., 'numpy', 'pandas', etc.
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    package_data={
        '': ['*.txt', '*.md', '*.json'],  # Include all txt, md, and json files
    },
    include_package_data=True
)
