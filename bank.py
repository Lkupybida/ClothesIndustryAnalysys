import openpyxl
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# banks = ["privatbank", "oschadbank", "ukrgasbank", "ukrexim", "sense", "first investment"]
banks = ["raiffeisen", "fuib", "ukrsibbank", "universal", "agricole", "otp", "citibank", "pivdenny", "kredobank", "procredit", "tascombank", "ing", "a - bank"]
def generate_date_range(start, end):
    """Generate a list of month-end dates from start to end."""
    date_range = pd.date_range(start=start, end=end, freq='M').strftime('%Y-%m').tolist()
    return date_range

def aggregate_data(column_name, output_csv, sheet_number):
    base_dir = 'original_dataset/aggregation'
    date_range = generate_date_range('2020-02', '2024-06')
    result_df = pd.DataFrame(index=date_range, columns=banks)
    
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if os.path.isdir(year_path):
            for month_file in os.listdir(year_path):
                month_path = os.path.join(year_path, month_file)
                if month_file.endswith('.xlsx'):
                    # Extract date from the filename and subtract one month
                    date_str = f"{year}-{month_file.split('-')[1]}"
                    date_obj = datetime.strptime(date_str, '%Y-%m')
                    date_obj -= relativedelta(months=1)
                    new_date_str = date_obj.strftime('%Y-%m')
                    
                    df = pd.read_excel(month_path, sheet_name=sheet_number)
                    df.columns = df.iloc[3]
                    if 1 in df.columns:
                        df.columns = df.iloc[2]
                    if df.empty or column_name not in df.columns:
                        continue
                    counter = -1
                    for bank in df["Bank"]:
                        if bank == "JSC 'ALFA-BANK'":
                            bank = "sense"
                        counter += 1
                        if not isinstance(bank, str):
                            continue
                        for bank_from_list in banks:
                            if bank_from_list in bank.lower():
                                result_df.at[new_date_str, bank_from_list] = df[column_name][counter]
                                break        
    # Move the last row to the first row
    result_df = result_df.iloc[[len(result_df)-1] + list(range(len(result_df)-1))]
    
    result_df.to_csv(output_csv)

def remove_rolling_sum(file_path, out_path):
    df = pd.read_csv(file_path, index_col=0)
    df_diff = df.diff(axis=0)
    for idx in df.index:
        if datetime.strptime(idx, '%Y-%m').month == 1:
            df_diff.loc[idx] = df.loc[idx]
    df_diff.to_csv(out_path)

# remove_rolling_sum('data/original/private_profit.csv', 'data/differenced/private_profit.csv')

aggregate_data('IGLB refinanced by NBU', 'private_iglb.csv', 0) 
