#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import io
import os
import subprocess
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

NAME = 'damei'
DESCRIPTION = 'Da Mei project with deep learning general functions.'
URL = 'https://github.com/zhangzhengde0225/damei'
EMAIL = 'drivener@163.com'
AUTHOR = 'Zhengde Zhang'
REQUIRES_PYTHON = '>=3.6.0'

with open('damei/version.py') as f:
    data = f.readlines()
VERSION = eval(data[0].split('=')[-1].strip())
print(f'Installing damei, version: {VERSION}')


def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


# Keep the base install dependency-free so `pip install damei` is fast and
# usable for lightweight utilities. Feature modules import optional
# dependencies only when they are used.
REQUIRED = []

EXTRAS = {
    'array': ['numpy'],
    'cv': ['numpy', 'opencv-python', 'tqdm'],
    'nn': ['numpy', 'easydict', 'torch', 'torchvision', 'pillow', 'tqdm'],
    'rsa': ['rsa'],
}
EXTRAS['all'] = sorted({pkg for deps in EXTRAS.values() for pkg in deps})

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    project_slug = NAME.lower().replace('-', '_').replace(' ', '_')
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Build, validate, upload, and tag a release."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for dirname in ('dist', 'build'):
            try:
                self.status(f'Removing previous {dirname}...')
                rmtree(os.path.join(here, dirname))
            except OSError:
                pass

        self.status('Building Source and Wheel (universal) distribution...')
        subprocess.check_call([sys.executable, 'setup.py', 'sdist', 'bdist_wheel', '--universal'])

        dists = sorted(glob.glob(os.path.join(here, 'dist', '*')))
        if not dists:
            raise RuntimeError('No distributions were built in dist/.')

        self.status('Checking distributions with Twine...')
        try:
            import twine  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                'Twine is required to publish. Install or upgrade publishing tools with: '
                '"python -m pip install -U twine pkginfo build wheel setuptools".'
            ) from e

        try:
            subprocess.check_call([sys.executable, '-m', 'twine', 'check'] + dists)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                'Twine failed to validate the distributions. If the error says metadata is missing '
                'Name/Version while PKG-INFO or METADATA contains them, upgrade the publishing tools: '
                '"python -m pip install -U twine pkginfo build wheel setuptools".'
            ) from e

        self.status('Uploading the package to PyPI via Twine...')
        subprocess.check_call([sys.executable, '-m', 'twine', 'upload'] + dists)

        self.status('Pushing git tag...')
        tag = f'v{about["__version__"]}'
        local_tag_exists = subprocess.call(
            ['git', 'rev-parse', '-q', '--verify', f'refs/tags/{tag}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) == 0
        if local_tag_exists:
            self.status(f'Local tag {tag} already exists; reusing it.')
        else:
            subprocess.check_call(['git', 'tag', tag])
        subprocess.check_call(['git', 'push', 'origin', tag])

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*', 'tests.*']),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    cmdclass={
        'upload': UploadCommand,
    },
)
