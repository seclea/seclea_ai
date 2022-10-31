********
Tutorial
********

Introduction
############

The seclea-ai package is used for tracking ML model development. This means that most of the
functions are designed to record and upload activities or data associated with your modelling.


Usage Guide
###########

There are two main steps to using the package:

- Before you begin modelling you will need to upload the dataset that you are working with. This will currently need to be a csv, a Pandas DataFrame or a list of csv's.

- After each time you train a new model you will upload it using ``SecleaAI.upload_training_run()``.

Example Notebook
################

.. include :: Getting_Started_with_Seclea.md
   :parser: myst_parser.sphinx_
