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
    plt.plot(df['Date'], top_values, color='b', label='Top Values Sum')
    plt.title(f'CR{number_of_banks}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return top_values




# equity to assets

df_equity_state = pd.read_csv('data/quarterly/total_equity_capital.csv', index_col=0)
df_equity_private = pd.read_csv('data/quarterly/private_equity_capital.csv', index_col=0)
df_assets_state = pd.read_csv('data/quarterly/total_assets.csv', index_col=0)
df_assets_private = pd.read_csv('data/quarterly/private_total_assets.csv', index_col=0)

numeric_cols = df_equity_state.select_dtypes(include=np.number).columns.tolist()
df_equity_state['nim_state'] = (df_equity_state[numeric_cols].sum(axis=1) - df_equity_state['privatbank']) / 5

numeric_cols_2 = df_equity_private.select_dtypes(include=np.number).columns.tolist()
df_equity_private['nim_private'] = df_equity_private[numeric_cols_2].sum(axis=1) / 13

numeric_cols = df_assets_state.select_dtypes(include=np.number).columns.tolist()
df_assets_state['assets_state'] = (df_assets_state[numeric_cols].sum(axis=1) - df_assets_state['privatbank']) / 5

numeric_cols_2 = df_assets_private.select_dtypes(include=np.number).columns.tolist()
df_assets_private['assets_private'] = df_assets_private[numeric_cols_2].sum(axis=1) / 13

df_assets = pd.DataFrame({
    'state': df_assets_state['assets_state'],
    'private': df_assets_private['assets_private'],
    'privatbank': df_assets_state['privatbank']
})
df_equity = pd.DataFrame({
    'state': df_equity_state['nim_state'],
    'private': df_equity_private['nim_private'],
    'privatbank': df_equity_state['privatbank']
})

start_date = '2023-01-01'
end_date = '2023-12-31'

df_final_nim = (df_equity / df_assets) * 100

filtered_df = df_final_nim.loc[start_date:end_date]


print(filtered_df)
print(filtered_df['state'].mean())
print(filtered_df['private'].mean())
print(filtered_df['privatbank'].mean())
