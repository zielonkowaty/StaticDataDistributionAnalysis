# importing dependencies

import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as matlib
import seaborn as seaborn
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display


seaborn.set_style("whitegrid")
matlib.rcParams['figure.figsize'] = (14,7)
seaborn.set_palette("muted")

# loading data from datasheets

file_path = '/Users/zielonkowaty/Documents/Pycharm/kagle_csv/instagram_usage_lifestyle.csv'
file_secondary_path = '/Users/zielonkowaty/Documents/Pycharm/kagle_csv/instagram_users_lifestyle.csv'

try:
    file = pandas.read_csv(file_path)
    print("File loaded")
except Exception as e:
    print("An error occurred while loading the file::", e)
    raise

try:
    file1 = pandas.read_csv(file_secondary_path)
    print("Second file loaded")
except Exception as e:
    print("An error occurred while loading the file:", e)


# print proporties
print("\nShape:", file.shape)
print("Columns:", file.columns.tolist())
display(file.head(5))
print("\nShape:", file1.shape)
print("Columns:", file1.columns.tolist())
display(file1.head(5))

# func to import data from columns

def fetch_column_data(file, possible_names):
    for name in possible_names:
        if name in file.columns:
            return name
    return None

# age & gender analysis

age_col = fetch_column_data(file, ['Age', 'age', 'User Age', 'user_age'])
gender_col = fetch_column_data(file, ['Gender', 'gender', 'Sex', 'sex'])

if age_col:
    matlib.figure()
    seaborn.histplot(file[age_col], bins=25, kde=True, color='green')
    median_age = file[age_col].median()
    matlib.axvline(median_age, linestyle='--', linewidth=2, label=f'Median: {median_age:.2f}')
    matlib.title(f'Age')
    matlib.legend()
    matlib.show()
else:
    print("Age column not found.")


if gender_col:
    matlib.figure()
    file[gender_col].value_counts().plot.pie(autopct='%1.1f%%', colors=seaborn.color_palette('pastel'))
    matlib.title(f'Gender')
    matlib.show()

# usage and lifestyle

time_col = fetch_column_data(file1, ['daily_active_minutes_instagram', 'Daily_Instagram_Time_minutes', 'Daily Usage Time', 'instagram_time',
'daily_time_minutes', 'Usage Time (minutes)', 'Time Spent'])

sleep_col = fetch_column_data(file1, ['sleep_hours_per_night', 'Sleep_Hours', 'sleep_hours', 'Sleep Hours', 'Avg Sleep'])

if time_col and sleep_col:
    matlib.figure()
    matlib.scatter(file1[time_col], file1[sleep_col], alpha=0.5)
    matlib.xlabel(time_col)
    matlib.ylabel(sleep_col)
    matlib.title('Time Spent on Instagram vs Sleep Time')
    matlib.show()
elif time_col:
    seaborn.histplot(file1[time_col], kde=True, color='teal')
    matlib.title(f'Daily Instagram Time Spent ({time_col})')
    matlib.show()
else:
    print("Columns not found")

# data correlation

numeric_file = file.select_dtypes(include=numpy.number)
if not numeric_file.empty:
    matlib.figure(figsize=(10,8))
    seaborn.heatmap(numeric_file.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    matlib.title('Correlation')
    matlib.show()