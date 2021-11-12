"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Setup.py for the PyRND project
"""
import setuptools
from os import path


here = path.abspath(path.dirname(__file__))
with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line
        for line in requirements_file.read().splitlines()
        if not line.startswith("#")
    ]


with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyRND",
    version="0.0.1",
    author="Samuel Tovey",
    author_email="tovey.samuel@gmail.com",
    description="A tool for performing random network distillation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zincware/PyRND.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
