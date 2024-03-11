# Retail-Data-Mining

This repository has 10 dataset files, 1 python file, and 1 jupyter notebook file, and 1 midtermm report pdf.

## Dataset Files:

amazonlist.csv
amazontransactions.csv
bestbuylist.csv
bestbuytransactions.csv
kmartlist.csv
kmarttransactions.csv
nikelist.csv
niketransactions.csv
generallist.csv
generaltransactions.csv

## Python file

brute_apriori_fpgrowth_algorithms.py

## Jupyter Notebook File

jupyter_midterm_code.ipynb

### The midterm report pdf

CS634 Midterm Project - Saketh Lakshmanan.pdf

### The libraries imported for the code are as follows:

import csv
import sys
import numpy as np
from itertools import chain, combinations
import pandas as pd
import time
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

## Use Jupyter Notebook to run this code as that is where this code was developed


