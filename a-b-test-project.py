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
#Check for any duplicate user id's
print(df2[df2['user_id'].duplicated(keep=False)])
#Drop one of the duplicated
df2.drop_duplicates('user_id', inplace=True)
#Now get the proportion of conversions for control and treatment
df_control = df2.query('group == "control"')
cont_convert = df_control.query('converted == 1').shape[0] / df_control.shape[0]

df_treat = df2.query('group == "treatment"')
treat_convert = df_treat.query('converted == 1').shape[0] / df_treat.shape[0]

obs_diff = treat_convert - cont_convert
#Now lets run an A/B test by simulating the null hypothesis
pnewnull = df2.query('converted == 1').shape[0] / df2.shape[0]
poldnull = df2.query('converted == 1').shape[0] / df2.shape[0]
nnew = df2.query('landing_page == "new_page"').shape[0]
nold = df2.query('landing_page == "old_page"').shape[0]

p_diffs = []
for _ in range(10000):
    new_convert = np.random.choice([0, 1], size=nnew, p=[1-pnewnull, pnewnull])
    old_convert = np.random.choice([0, 1], size=nold, p=[1-poldnull, poldnull])
    p_diffs.append(new_convert.mean() - old_convert.mean())

p_diffs = np.asarray(p_diffs)
#Plot the histogram of the differences and the observed difference as a red line
plt.hist(p_diffs)
plt.axvline(obs_diff, color='r')
plt.savefig('differencehistogram.png')
plt.clf()

pvalue = (p_diffs > obs_diff).mean()
print(pvalue)
