import pandas as pd
import csv

import Data_preprocessing as pre
import Bug_report_processed as brp
import Extract_feature_vector as efv

# convert txt to csv
with open('Training_Test\AspectJ.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open('Training_Test\AspectJ_csv.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)

# Read file csv and Extract summary and description
inp = pd.read_csv('Training_Test\AspectJ_csv.csv')
x= inp[['bug_id', 'summary']]
df =pd.DataFrame(x)
df.columns = df.columns.str.replace('summary','description')
df.columns = df.columns.str.replace('bug_id','summary')
df.to_csv('Training_Test\AspectJ_process.csv')

#Data pre-processing
pre.Start()
#
#Word_Embedding
brp.word_embedding()

#Extract feature vector
efv.Start()








