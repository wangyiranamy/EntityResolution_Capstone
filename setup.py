from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='entity_resolver',
    version='0.1.1',
    author='Yiran Wang',
    author_email='yw3201@columbia.edu',
    description='A python package for collective entity resolution',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wangyiranamy/EntityResolution_Capstone',
    package_dir={'': 'src'},
    packages=[
        'entity_resolver', 'entity_resolver.core', 'entity_resolver.parser'
    ],
    entry_points={
        'console_scripts': [
            'entity-resolver=entity_resolver.scripts:main',
        ]
    },
    python_requires='>=3.6',
    install_requires=[
        'pandas>=0.25.1', 'numpy>=1.17.2', 'scikit-learn>=0.21.3',
        'nltk>=3.4.5', 'py-stringmatching>=0.4.1', 'matplotlib>=3.1.2'
    ]
)
