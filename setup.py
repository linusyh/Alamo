from setuptools import setup, find_packages

setup(
    name='Alamo',
    version='0.1.0',
    author='Linus Leong',
    author_email='yhll2@cam.ac.uk',
    description='A framework for federated and split learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/alamo',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # e.g., 'requests>=2.20',
    ],
    classifiers=[
        # Choose your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)