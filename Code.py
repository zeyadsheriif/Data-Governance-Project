!pip install dataprofiler
!pip install pandera
!pip install anonymizedf

import json
import numpy as np
import pandas as pd
from dataprofiler import Data, Profiler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandera as pa
import random
import string
from pandera import Column, DataFrameSchema, Check, Index
import os
from anonymizedf.anonymizedf import anonymize
import hashlib
import pandera as pa
from pandera import Column, DataFrameSchema, Check, Index

df = pd.read_csv("/content/Indicators_of_Health_Insurance_Coverage_at_the_Time_of_Interview.csv")

df.head()

host = df.sort_values(by='Value')
host.head()

df.describe()

df.shape

df.duplicated().sum()

df.isnull().sum()

df.describe()

# Data Profiling:
# Profile the dataset
profile = Profiler(df) # Calculate Statistics, Entity Recognition, etc
# Generate a report and use json to prettify.
report  = profile.report(report_options={"output_format":"pretty"})
# Print the report
print(json.dumps(report, indent=4))
# Print the report using json to prettify.
report  = profile.report(report_options={"output_format":"pretty"})
print(json.dumps(report, indent=4))

df.info()

# Data Cleaning:
df.sample(5)

duplicate = df[df.duplicated()]
duplicate

df.drop_duplicates()

missing_values_count = df.isnull().sum()

# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()
# percent of data that is missing
(total_missing/total_cells) * 100

# remove all the rows that contain a missing value
df.dropna()

df = df.drop('Suppression Flag', axis=1)

# get the number of missing data points per column
missing_values_count = df.isnull().sum()
# look at the # of missing points in the first ten columns
missing_values_count[:]

A = df['Value'].mean()
df['Value'].fillna(A, inplace = True)

B = df['Low CI'].median()
df['Low CI'].fillna(B, inplace = True)

C = df['High CI'].median()
df['High CI'].fillna(C, inplace = True)

df['Confidence Interval'].fillna(df['Confidence Interval'].mode()[0], inplace=True)

df['Quartile Range'].fillna(df['Quartile Range'].mode()[0], inplace=True)

F = df['Quartile Number'].median()
df['Quartile Number'].fillna(F, inplace = True)

# get the number of missing data points per column
missing_values_count = df.isnull().sum()
# look at the # of missing points in the first ten columns
missing_values_count[:]

df['Value'] = df['Value'].astype(float)
df['Low CI'] = df['Low CI'].astype(float)
df['High CI'] = df['High CI'].astype(float)
df['Quartile Number'] = df['Quartile Number'].astype(float)

df.dtypes

numerical_columns=['Value', 'Low CI', 'High CI', 'Quartile Number']
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3-Q1
print('IQR Scores of columns:\n',IQR)


# Data validation:
schema = pa.DataFrameSchema(
    {
        "Indicator": Column(object),
        "Group": Column(object),
        "State": Column(object),
        "Subgroup": Column(object),
        "Phase": Column(object),
        "Time Period": Column(int),
        "Time Period Label": Column(object),
        "Time Period Start Date": Column(object),
        "Time Period End Date": Column(object),
        "Value": Column(float),
        "Low CI": Column(float),
        "High CI": Column(float),
        "Confidence Interval": Column(object),
        "Quartile Range": Column(object),
        "Quartile Number": Column(float),

    },
)

validated_df = schema(df)
print(validated_df)

try:
    schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as exc:
    failure_cases_df = exc.failure_cases
    display(exc.failure_cases)

# Data Security and encryption:
validated_df

bins = [0, 22, 42, 62, 82, 92]
labels = ['0-22', '42-62', '62-82', '82-92', '92+']
validated_df['Value'] = pd.cut(validated_df['Value'], bins=bins, labels=labels)

validated_df

an = anonymize(validated_df)

an.fake_names("State")
an.fake_names("Group")
an.fake_names("Subgroup")
an.fake_categories("Indicator")
an.fake_whole_numbers("Time Period")
validated_df

validated_df['Hashed_Time Period Label'] = validated_df['Time Period Label'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
validated_df

# Role-Based Access Control (RBAC)

roles = ['Data_Viewer', 'Data_editor', 'IT_manager', 'adminstrator']
permissions = ['View Data', 'View Secured Data', 'Edit the data', 'Transfer the data']


rbac_df = pd.DataFrame(columns=['Role'] + permissions)
rbac_df['Role'] = roles
rbac_df.set_index('Role', inplace=True)


rbac_df.loc['IT_manager', ['Transfer the data']] = 1
rbac_df.loc['adminstrator', ['View Data', 'View Secured Data', 'Edit the data', 'Transfer the data']] = 1
rbac_df.loc['Data_Viewer', 'View Data'] = 1
rbac_df.loc['Data_editor', ['View Data', 'View Secured Data', 'Edit the data']] = 1

rbac_df = rbac_df.fillna(0)

rbac_df

validated_df.to_csv('validated_df.csv',index=False)

