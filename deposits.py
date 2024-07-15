import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# banks = ["приватбанк", "ощадбанк", "укргазбанк", "укрексім", "сенс", "перший інвестиційний"]
banks = ["райффайзен", "укрсиббанк", "південний", "кредобанк", "універсал", "таскобанк", "а-банк", "сітібанк", "агріколь", "отп", "прокредит", "інг"]
def generate_date_range(start, end):
    """Generate a list of month-end dates from start to end."""
    date_range = pd.date_range(start=start, end=end, freq='M').strftime('%Y-%m').tolist()
    return date_range

def aggregate_data(output_csv):
    date_range = generate_date_range('2020-02', '2024-06')
    result_df = pd.DataFrame(index=date_range, columns=banks)
    # Load the Excel file
    xls = pd.ExcelFile('FG_2024-06-01.xlsx')

    # Iterate over all sheet names
    for sheet_name in xls.sheet_names[12:]:
        current_date_list = sheet_name.split('.')
        current_date = f'{current_date_list[2]}-{current_date_list[1]}'
        date_obj = datetime.strptime(current_date, '%Y-%m')
        date_obj -= relativedelta(months=1)
        new_date_str = date_obj.strftime('%Y-%m')
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[1, 4], skiprows=6)
        df.columns = ['Bank', 'Deposit']
        if 2 in df['Bank'].values:
            df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[2, 5], skiprows=6)
            df.columns = ['Bank', 'Deposit']
            print(df)
        # Do something with the DataFrame
        for index, row in df.iterrows():
            for bank in banks:
                if not isinstance(row["Bank"], str):
                    continue
                if "АЛЬФА" in row["Bank"]:
                    row["Bank"] = "сенс"
                if bank in row["Bank"].lower():
                    result_df.at[new_date_str, bank] = row["Deposit"]
                    break        
    result_df = result_df.iloc[[len(result_df)-1] + list(range(len(result_df)-1))]
    result_df.to_csv(output_csv)
   

# aggregate_data("private_deposit.csv")
    

# Load the data
df = pd.read_csv('data/quarterly/deposit.csv')
df.columns = ["date", "приватбанк", "ощадбанк", "укргазбанк", "укрексім", "сенс", "перший інвестиційний"]
banks = ["приватбанк", "ощадбанк", "укргазбанк", "укрексім", "сенс", "перший інвестиційний"]

# Initialize an empty dictionary to store results
result_data = {bank: [] for bank in banks}
result_data['date'] = []

# Process each row
for _, row in df.iterrows():
    result_data['date'].append(row['date'])
    for bank in banks:
        if row['date'] in ['2024-06-30']:
            result_data[bank].append((row[bank] / 2) * 3)
        else:
            result_data[bank].append(row[bank])

# Convert the result dictionary to a DataFrame
result_df = pd.DataFrame(result_data)

# Save the result to a CSV file
result_df.set_index('date', inplace=True)
result_df.to_csv('data/quarterly/deposit.csv')