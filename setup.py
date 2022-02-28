from setuptools import setup
from os import path

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements


# Open the peripheral files and load into ram
installReqs = [n.requirement for n in parse_requirements('requirements.txt', session='hack')]
longdesc = None
with open('./README.md', 'r') as handle:
    longdesc = handle.read()
libpath:str = f'bin{path.sep}'

# Main setuptools initialization
setup(
    name='PlasmaTorch',
    version='0.1',
    description='A continuous signal ML framework built on torch.',
    long_description=longdesc,
    long_description_content_type='text/markdown',
    author='labrat97',
    author_email='james@nineseven.net',
    url='https://github.com/labrat97/plasmatorch',
    packages=['plasmatorch'],
    install_requires=installReqs
)