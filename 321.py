import random

import pandas as pd
import openpyxl

russian = openpyxl.load_workbook('knowledge\\Russian.xlsx')
hohli = pd.read_excel('knowledge\\hohli.xlsx')

russians = russian.active
pos = random.uniform(-1, 1)

print(type(russians.cell(row=5, column=4).value))
print(pos)