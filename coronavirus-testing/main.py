import pandas as pd
import matplotlib.pyplot as plt

ALL_DATA_FILE = './owid-covid-data.csv'
TEST_DATA_FILE = './covid-testing-all-uk.csv'

UK= 'United Kingdom'

df = pd.read_csv(ALL_DATA_FILE)

df = df[df['location'] == UK]
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df.sort_values(by=['date'])

df['test_vs_cases'] = df['new_cases'] * (df['new_cases'] / df['new_tests'])
print(df.head(10))

plt.rcParams["figure.figsize"] = (12,6)
plt.xticks(rotation=90)
# plot cases
df = df[df['date'] > pd.to_datetime('2020-6-01')]
df['test_vs_cases'].plot()
df['new_cases'].plot()
plt.show()
print("[*] Plot")
