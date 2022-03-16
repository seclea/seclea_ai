# -*- coding: utf-8 -*-

__doc__ = f""" There are all configuration of the project.

    The settings.py is the central configuration for the project.
    In previous chapters you already worked with a series of variables in this
    file to configure things like applications, databases, templates and
    middleware, among other things.

    Although the settings.py file uses reasonable default values for
    practically all variables, when a application transitions into the
    real world, you need to take into account a series of adjustments, to
    efficiently run the application, offer end users a streamlined
    experience and keep potentially rogue attackers in check.

    The way application is designed usually requires the configuration to be
    available when the application starts up. You can hard code the
    configuration in the code, which for many small applications is not
    actually that bad, but there are better ways.

    Independent of how you load your config, there is a config object available
    which holds the loaded configuration values: The config attribute of the
    Application object. This is the place where Application itself puts certain
    configuration values and also where extensions can put their configuration
    values. But this is also where you can have your own configuration.



"""


#
# import the base pachage data
#
from src import (
    # Package version
    __version__,

    # import the package
    __package__,
)
__name__ = 'settings.base'
__date__ = 'Feb 25, 2022'


#
# Inform the application what path it is mounted under by the application / src
# server. This is used for generating URLs outside the context of a request
# (inside a request, the dispatcher is responsible for setting SCRIPT_NAME
# instead; see Application Dispatching for examples of dispatch
# configuration).
#
#    Default: '/'
#
def __BASE__():
    #
    # The following documentation is automatically generated from the Python
    # source files. It may be incomplete, incorrect or include features that
    # are considered implementation detail and may vary between Python
    # implementations. When in doubt, consult the module reference at the
    # location listed above. Read more about the lib: https://cutt.ly/SR0FI2T
    #
    return __import__('pathlib').Path().absolute()


def __DATA__(): return __BASE__() / 'data'


#
# Since the config object provided loading of configuration files from relative
# filenames we made it possible to change the loading via filenames to be
# relative to the instance path if wanted. The behavior of relative paths in
# config files can be flipped between “relative to the application root” (the
# default) to “relative to instance folder” via the instance_relative_config
# switch to the application constructor:
#
def __PATH__(path: str) -> 'pathlib.PosixPath':
    """
        A path is a string of characters used to uniquely identify a location
        in a directory structure. It is composed by following the directory
        tree hierarchy in which components, separated by a delimiting
        character, represent each directory.

        The delimiting character is most commonly the slash ("/"), the
        backslash character ("\\": double slashes equals to one slash),
        or colon (":"), though some operating systems may use a different
        delimiter. Paths are used extensively in computer science to represent
        the directory/file relationships common in modern operating systems,
        and are essential in the construction of Uniform Resource Locators
        (URLs). Resources can be represented by either absolute or relative
        paths.


        paths:
            {
              'ROOT_PATH': /,
              "DOCS": __DATA__() / 'docs',
              'MEDIA': __DATA__() / 'media',
              'EXAMPLES': __DATA__() / 'docs/code/examples',

              ...

            }
    """

    __path__ = {
        "DOCS": __DATA__() / 'docs',
        'MEDIA': __DATA__() / 'media',
        'EXAMPLES': __DATA__() / 'docs/code/examples',
    }

    if path.upper() in __path__.keys():
        return __path__[path.upper()]

    else:
        return f"You need to choose from the {list(__path__.keys())} list"


#
# SECRET_KEY: A secret key that will be used for securely signing the session
#             cookie and can be used for any other security related needs by
#             extensions or your application. It should be a long random bytes
#             or str. For example, copy the output of this to your config:
#
def __SECRET_KEY__(bit=512):
    #
    # https://docs.python.org/3.10/library/secrets.html
    #
    # The following documentation is automatically generated from the Python
    # source files.  It may be incomplete, incorrect or include features that
    # are considered implementation detail and may vary between Python
    # implementations.  When in doubt, consult the module reference at the
    # location listed above.
    #
    return __import__('secrets').token_hex(bit)
