import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Subject': ['Math', 'Math', 'Math', 'Science', 'Science', 'English', 'English', 'History', 'History', 'History'],
    'Score': np.random.randint(50, 101, 10),
    'Grade': [''] * 10
}
df = pd.DataFrame(data)

def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['Grade'] = df['Score'].apply(assign_grade)

sorted_df = df.sort_values(by='Score', ascending=False)
print("DataFrame sorted by Score (descending):")
print(sorted_df)

average_scores = df.groupby('Subject')['Score'].mean()
print("\nAverage score for each subject:")
print(average_scores)

def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

passed_df = pandas_filter_pass(df.copy())
print("\nDataFrame with grades A or B:")
print(passed_df)