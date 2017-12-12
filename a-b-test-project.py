import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)

ab_data = pd.read_csv('ab_data.csv')

#See when new_page and treatment don't line up
treat = ab_data.query('group == "treatment"')
contr = ab_data.query('group == "control"')
print(contr['landing_page'].value_counts(), treat['landing_page'].value_counts())

#Create a cleaned dataframe
treat2 = treat[treat['landing_page'] == 'new_page']
contr2 = contr[contr['landing_page'] == 'old_page']

df2 = treat2.append(contr2)