import random

import pandas as pd
import openpyxl

russian = openpyxl.load_workbook('knowledge\\Russian.xlsx')
hohli = openpyxl.load_workbook('knowledge\\hohli.xlsx')

russians = russian.active
h = hohli.active
pos = random.uniform(-1, 1)

def func():
    rus_dol_year = 11600
    ucr_dol_year = 2724
    rus = 15
    hohl = 15
    T = 0.25
    one_one = 13
    return (rus_dol_year / ucr_dol_year) * (hohl / rus) * T * one_one

print(func())