#!/usr/bin/env python
# -*- coding: utf-8 -*-

# setup.py
#
# The setup script is the centre of all activity in building, distributing and
# installing modules using the Distutils, so that the various commands that
# operate on your modules do the right thing.
# See https://cutt.ly/smbaTl8
#
# Copyright (c) 2022 Melkon Hovhannisyan <Melkon.Hovhannisian@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

""" A setuptools based setup module.

setup.py is a python file, which usually tells you that the module/package you
are about to install has been packaged and distributed with Distutils, which is
the standard for distributing Python Modules.


See:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
    https://github.com/pypa/sampleproject
"""


#
# https://docs.python.org/3.10/library/pathlib.html
#
# The following documentation is automatically generated from the Python
# source files.  It may be incomplete, incorrect or include features that
# are considered implementation detail and may vary between Python
# implementations.  When in doubt, consult the module reference at the
# location listed above.
#
import pathlib

#
# Always prefer setuptools over distutils
#
# Return a list all Python packages found within directory 'where'
#
# 'where' is the root directory which will be searched for packages.  It
# should be supplied as a "cross-platform" (i.e. URL-style) path; it will
# be converted to the appropriate local path syntax.
#
# 'exclude' is a sequence of package names to exclude; '*' can be used
# as a wildcard in the names, such that 'foo.*' will exclude all
# subpackages of 'foo' (but not 'foo' itself).
#
# 'include' is a sequence of package names to include.  If it's
# specified, only the named packages will be included.  If it's not
# specified, all found packages will be included.  'include' can contain
# shell style wildcard patterns just like 'exclude'.
#
from setuptools import setup, find_packages


__AUTHOR__ = str(__import__('src.__version__').__author__).split(' <')
__author__ = __AUTHOR__[0]
__author_email__ = __AUTHOR__[1].strip('>')
__NAME__ = __import__('src.__version__').__name__
__PACKAGE__ = __import__('src.__version__').__package__


#
# PurePath subclass that can make system calls.
#
# Path represents a filesystem path but unlike PurePath, also offers
# methods to do system calls on path objects. Depending on your system,
# instantiating a Path will return either a PosixPath or a WindowsPath
# object. You can also instantiate a PosixPath or WindowsPath directly,
# but cannot instantiate a WindowsPath on a POSIX system or vice versa.
#
from src.core.settings import __PATH__

__FILE__ = __PATH__('DOCS') / 'Seclea/about.txt'


#
# Get the long description from the docs of Warehouse.
#
long_description = (__FILE__).read_text(encoding='utf-8')


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install __PACKAGE__
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name=__NAME__,  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/guides/single-sourcing-package-version/
    version=__import__('src.__version__').__version__,          # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    #
    # Optional
    description=""" \vThis is an example:

    Websites are designed to cater to people's strengths. Humans have an
    incredible ability to take visual information, combine it with our
    experiences to derive meaning, and then act on that meaning. It's why you
    can look at a form on a website and know that the little box with the
    phrase "First Name" above it means you are supposed to type in the word you
    use to informally identify yourself.

    Yet, what happens when you face a very time-intensive task, like copying
    the contact info for a thousand customers from one site to another? You
    would love to delegate this work to a computer so it can be done quickly
    and accurately. Unfortunately, the characteristics that make websites
    optimal for humans make them difficult for computers to use.

    The solution is an API. An API is the tool that makes a website's data
    digestible for a computer. Through it, a computer can view and edit data,
    just like a person can by loading pages and submitting forms.

    When talking about APIs, a lot of the conversation focuses on abstract
    concepts. To anchor ourselves, let's start with something that is physical:
    the server. A server is nothing more than a big computer. It has all the
    same parts as the laptop or desktop you use for work, it’s just faster and
    more powerful. Typically, servers don't have a monitor, keyboard, or mouse,
    which makes them look unapproachable. The reality is that IT folks connect
    to them remotely — think remote desktop-style — to work on them.

    Servers are used for all sorts of things. Some store data; others send
    email. The kind people interact with the most are web servers. These are
    the servers that give you a web page when you visit a website.
    """,

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://cutt.ly/dUjvlAG
    #
    # Optional
    long_description=f"\n{long_description}",

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://cutt.ly/dUjvG37
    long_description_content_type='text/plain',  # Optional (see note above)

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://cutt.ly/HUjvZzD
    url='https://pypi.org/project/seclea-ai/',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    #
    # Optional
    author=__author__,

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email=__author_email__,  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        # extras_rtional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: GPL3 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3 :: Only',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a list of additional keywords, separated
    # by commas, to be used to assist searching for the distribution in a
    # larger catalog.
    #
    # Optional
    keywords='login, sample, auth, setuptools, development',

    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={__NAME__: 'src/'},  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    # packages=find_packages(where=__NAME__),  # Required
    packages=[
        __NAME__,
    ],  # Required
    # py_modules=[__NAME__],

    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://cutt.ly/AUjv1VU
    python_requires='>=3.6, <4',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://cutt.ly/0UU19cY
    #
    # Optional
    install_requires=[
       'seclea-utils',
    ],

    #
    # A dictionary mapping package names to lists of filenames
    # or globs to use to find data files contained in the named packages.
    # If the dictionary has filenames or globs listed under '""' (the empty
    # string), those names will be searched for in every package, in addition
    # to any names for the specific package.  Data files found using these
    # names/globs will be installed along with the package, in the same
    # location as the package.  Note that globs are allowed to reference
    # the contents of non-package subdirectories, as long as you use '/' as
    # a path separator. (Globs are automatically converted to
    # platform-specific paths at runtime.)
    #
    platforms=['Linux', 'FreeBSD'],
    license='GPLv3',

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install WebSite[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        'dev': ['check-manifest', 'pep8', 'autopep8', 'ipython'],
        'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    include_package_data=True,
    package_data={  # Optional
        f'{__PACKAGE__}': ['package_data.dat'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # https://cutt.ly/sUjv9sG
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/data'
    data_files=[('data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invokedv
    entry_points={  # Optional
        'console_scripts': [
            f'{__PACKAGE__}=src:main',
        ],
    },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://cutt.ly/NUjnUJa
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/pypa/sampleproject/issues',

        'Documentation': 'https://cutt.ly/jUjnSMq',
        'Funding': 'https://donate.pypi.org',
        'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/pypa/sampleproject/',
        'Tracker': 'https://github.com/pypa/sampleproject/issues',
    },
)
