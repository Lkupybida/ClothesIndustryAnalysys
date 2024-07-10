import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_CR(number_of_banks):
    df = pd.read_csv('data/original/TA.csv')

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df['total'] = df[numeric_cols].sum(axis=1)

    for i in range(len(df)):
        total_value = df.at[i, 'total']
        for col in numeric_cols:
            df.at[i, col] = df.at[i, col] / total_value

    top_values = []

    for i in range(len(df)):
        row = df.iloc[i]
        sorted_row = row[numeric_cols].sort_values(ascending=False)
        sum_of_top_values = sorted_row[:number_of_banks].sum()
        top_values.append(sum_of_top_values)

    date_range = pd.date_range(start='2020-01-01', end='2024-04-01', freq='M')

    df['Date'] = date_range
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], top_values, marker='o', linestyle='-', color='b', label='Top Values Sum')
    plt.title('Sum of Top Values vs. Dates')
    plt.xlabel('Date')
    plt.ylabel('Sum of Top Values')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return top_values

print(calculate_CR(3))
