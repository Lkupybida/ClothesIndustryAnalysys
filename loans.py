import openpyxl
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# banks = ["приватбанк", "ощадбанк", "укргазбанк", "укрексім", "сенс", "перший інвестиційний"]
banks = ["райффайзен", "укрсиббанк", "південний", "кредобанк", "універсал", "таскомбанк", "а-банк", "сітібанк", "агріколь", "отп", "прокредит", "інг"]
def generate_date_range(start, end):
    """Generate a list of month-end dates from start to end."""
    date_range = pd.date_range(start=start, end=end, freq='M').strftime('%Y-%m').tolist()
    return date_range

def aggregate_data(output_csv):
    base_dir = 'original_dataset/Loans_KVED(1)'
    date_range = generate_date_range('2020-01', '2024-06')
    result_df = pd.DataFrame(index=date_range, columns=banks)
    
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if os.path.isdir(year_path):
            for month_file in os.listdir(year_path):
                month_path = os.path.join(year_path, month_file)
                if month_file.endswith('.xlsx'):
                    # Extract date from the filename and subtract one month
                    date_str = f"{month_file.split('_')[2][:4]}-{month_file.split('-')[1]}"
                    print(date_str)
                    date_obj = datetime.strptime(date_str, '%Y-%m')
                    date_obj -= relativedelta(months=1)
                    new_date_str = date_obj.strftime('%Y-%m')
                    
                    df = pd.read_excel(month_path, skiprows=4, usecols=[1, 4])
                    df.columns = ['Bank', 'Loan']
                    
                    for _, row in df.iterrows():
                        if row['Bank'] == '272 АТ "АЛЬФА-БАНК"':
                            row['Bank'] = "сенс"
                        if not isinstance(row['Bank'], str):
                            continue
                        for bank_from_list in banks:
                            if bank_from_list in row['Bank'].lower():
                                if pd.isna(result_df.at[new_date_str, bank_from_list]):
                                    result_df.at[new_date_str, bank_from_list] = row['Loan']
                                else:
                                    result_df.at[new_date_str, bank_from_list] += row['Loan']
                                break

    # result_df = result_df.iloc[[len(result_df)-1] + list(range(len(result_df)-1))]
    
    result_df.to_csv(output_csv)



# aggregate_data('private_loans.csv')


# Load the data
df = pd.read_csv('data/quarterly/private_loans.csv')
df.columns = ["date", "райффайзен", "укрсиббанк", "південний", "кредобанк", "універсал", "таскомбанк", "а-банк", "сітібанк", "агріколь", "отп", "прокредит", "інг"]
banks = ["райффайзен", "укрсиббанк", "південний", "кредобанк", "універсал", "таскомбанк", "а-банк", "сітібанк", "агріколь", "отп", "прокредит", "інг"]

# Initialize an empty dictionary to store results
result_data = {bank: [] for bank in banks}
result_data['date'] = []

# Process each row
for _, row in df.iterrows():
    result_data['date'].append(row['date'])
    for bank in banks:
        if row['date'] in ['2020-12-31', '2022-03-31', '2022-06-30', '2024-06-30']:
            result_data[bank].append((row[bank] / 2) * 3)
        else:
            result_data[bank].append(row[bank])

# Convert the result dictionary to a DataFrame
result_df = pd.DataFrame(result_data)

# Save the result to a CSV file
result_df.set_index('date', inplace=True)
result_df.to_csv('data/quarterly/new_private_loans.csv')

