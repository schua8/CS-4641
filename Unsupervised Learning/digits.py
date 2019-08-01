# import gzip
# import csv
# import json
# import pandas as pd


# df = pd.read_csv('digits.csv.gz', compression='gzip', header=0,    sep=' ', quotechar='"', error_bad_lines=False)


# with open("digits.csv",'wt') as file:
#         writer = csv.writer(file, delimiter=',')
#         for row in df.iterrows():
#             writer.writerow(row['0'])
# for i in range (100):
# print df.next()

import gzip    

with gzip.open("digits.csv.gz", 'rt') as f:
	data = f.read()
with open("digits.csv", 'wt') as f:
    f.write(data)