import pandas as pd
import matplotlib.pyplot as plt

def sum_npl(file1, file2, bankname):
    df1 = pd.read_csv(file1, index_col='Date')
    df1['Total'] = df1.sum(axis=1, skipna=True)
    columns_to_drop = [f'{i:02d}' for i in range(100)]
    df1 = df1.drop(columns=columns_to_drop, errors='ignore')


    df2 = pd.read_csv(file2, index_col='Date')
    df2['Total'] = df2.sum(axis=1, skipna=True)
    columns_to_drop = [f'{i:02d}' for i in range(100)]
    df2 = df2.drop(columns=columns_to_drop, errors='ignore')
    result_data = df2.div(df1)

    result_data.index = pd.to_datetime(result_data.index)
    quarterly_df = result_data.resample('Q').mean()


    plt.figure(figsize=(10, 6))
    plt.plot(quarterly_df, marker='o', linestyle='-', color='#90BE6D', label=bankname)
    plt.title('Квартальний показник NPL Ratio для банку ' + bankname)
    plt.xlabel('Date')
    plt.ylabel('NPL Ratio')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return quarterly_df

file1 = 'data/loans/grouped/loans/PrivatBank.csv'
file2 = 'data/loans/grouped/npl/PrivatBank.csv'
sum_npl(file1, file2, "приват")