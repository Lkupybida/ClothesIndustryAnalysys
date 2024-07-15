import os
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import openpyxl

def extract_bank_data(root_folder, sheet_name, column, file):
    target_banks = ["privatbank", "oschadbank", "ukreximbank", "ukrgasbank", "alfa", "sense", "first investment bank"]
    data = {}

    for year in range(2020, 2025):
        folder = os.path.join(root_folder, str(year))
        if not os.path.exists(folder):
            continue

        for file_name in os.listdir(folder):
            if file_name.startswith("aggregation_") and file_name.endswith(".xlsx"):
                file_path = os.path.join(folder, file_name)
                date = datetime.strptime(file_name[12:22], "%Y-%m-%d")

                try:
                    # Try reading with 4th row as header
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=3)
                    target_col = find_target_column(df, column)

                    # If target column not found, try with 5th row as header
                    if target_col is None:
                        df = pd.read_excel(file_path, sheet_name=sheet_name, header=4)
                        target_col = find_target_column(df, column)

                    if target_col is None:
                        print(f"Warning: Target column '{column}' not found in {file_name}")
                        continue

                except Exception as e:
                    print(f"Error reading {file_name}: {str(e)}")
                    continue

                for bank in target_banks:
                    bank_row = df[df['Bank'].astype(str).str.lower().str.contains(bank, case=False, na=False)]
                    if not bank_row.empty:
                        value = bank_row[target_col].values[0]
                        if date not in data:
                            data[date] = {}
                        data[date][bank] = value

    result_df = pd.DataFrame.from_dict(data, orient='index')
    result_df.index.name = 'Date'
    result_df.sort_index(inplace=True)

    output_file = file + ".csv"
    result_df['sense'] = result_df['alfa'] + result_df['sense']
    result_df = result_df.drop(columns=['alfa'])
    result_df.to_csv(output_file)
    result_df.index = pd.to_datetime(result_df.index)


    # Shift all dates one month back
    result_df.index = result_df.index - DateOffset(months=1)
    print(f"Data extracted and saved to {output_file}")

def find_target_column(df, column):
    for col in df.columns:
        if isinstance(col, str) and column.lower() in col.lower():
            return col
    return None

# extract_bank_data('original_dataset/aggregation', 'Assets', 'Total assets', 'data/Total_Assets')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_donut_chart(csv_file, period, all_banks):
    # Load the CSV data
    df = pd.read_csv(csv_file, header=None)

    # Rename the first column to 'Period'
    df.columns = ['Period'] + df.iloc[0, 1:].tolist()
    df = df[1:]

    # Filter the data for the specified period
    df_filtered = df[df['Period'] == period]

    if df_filtered.empty:
        print(f"No data available for the specified period: {period}")
        return

    # Extract bank names and their values
    bank_names_1 = df.columns[1:]
    values = df_filtered.iloc[0, 1:].astype(float).tolist()  # Convert to list

    # Add the 'others' category
    bank_names_1 = list(bank_names_1) + ['others']
    values.append(3433067383.21959 - 1921129277)

    # Create a dictionary for bank colors
    all_all_banks = all_banks
    all_all_banks.append(['others', 'grey'])
    bank_colors = {bank_1[0]: bank_1[1] for bank_1 in all_all_banks}

    # Get the colors in the order of the bank names and convert them to RGBA with 50% transparency
    colors = [mcolors.to_rgba(bank_colors.get(bank_1, 'gray'), alpha=0.5) for bank_1 in bank_names_1]

    # Create the donut chart
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = ax.pie(values, autopct='%1.1f%%', startangle=140, pctdistance=0.85, colors=colors)

    # Draw a circle at the center to create a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Add labels
    ax.legend(wedges, bank_names_1, title="Banks", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight="bold")

    plt.title(f"Bank Values for Period: {period}")
    plt.show()
    all_banks = [['privatbank', 'lime'], ['oschadbank', 'black'], ['sense', 'blue'],
                 ['ukreximbank', 'magenta'], ['ukrgasbank', 'red'], ['first investment bank', 'orange']]
    df_filtered.to_csv('data/market_share/df_filtered.csv', index=False)


def plot_stacked_area_chart(csv_file, list):
    # Read CSV into pandas DataFrame
    df = pd.read_csv(csv_file)

    # Set the index to the first column (date) for easier plotting
    df.set_index(df.columns[0], inplace=True)
    df.index = pd.to_datetime(df.index)

    # Plotting
    plt.figure(figsize=(14, 7))

    # Plot the stacked area with transparency
    for bank in list:
        if bank[0] == 'sense':
            plt.fill_between(df.index, df[bank[0]], color=bank[1], label=bank[0])
        elif bank[0] == 'others':
            continue
        else:
            plt.fill_between(df.index, df[bank[0]], color=bank[1], alpha=0.5, label=bank[0])
        plt.plot(df.index, df[bank[0]], color=bank[1], linewidth=3)

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('Stacked Area Chart of Bank Balances')
    plt.legend()

    # Show plot
    plt.show()

    # Show plot
    plt.show()


def get_market_share(csv_file):
    df = pd.read_csv(csv_file)
    result = pd.DataFrame()

    # Assuming the first column is 'date'
    result['date'] = df.iloc[:, 0]

    # Remove the first column from df to only have numerical data
    df = df.drop(columns=df.columns[0])

    # Convert columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Calculate total for each row, ignoring NaN values
    df['total'] = df.sum(axis=1, skipna=True)

    # Calculate market share for each column
    for col in df.columns[:-1]:  # exclude the 'total' column
        result[col] = df[col] / df['total']

    result.to_csv('data/market_share/TA.csv', index=False)


get_market_share('data/original/TA_copy.csv')
import csv
import math
import logging

logging.basicConfig(level=logging.INFO)
def plot_HHI(file_path):
    result = []

    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            # Convert values to float, square them, and sum
            # Skip the first column (date) and any empty strings
            squared_sum = sum(float(val) ** 2 for val in row[1:] if val.strip())
            result.append(squared_sum)

    result_df = pd.DataFrame()
    df = pd.read_csv(file_path)
    result_df['date'] = df.iloc[:, 0]
    result_df['HHI'] = result

    result_df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    # result_df.set_index(df.iloc[:, 0], inplace=True)
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-poster')
    plt.title(os.path.splitext('HHI')[0])
    plt.plot(result_df.iloc[:, 0], result_df['HHI'])
    plt.legend()
    plt.grid(visible=False)
    plt.show()

def plot_theil(file_path):
    result = []

    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row

        for row_num, row in enumerate(csv_reader, start=2):  # Start from 2 to account for header
            try:
                # Convert values to float, multiply by log of itself, and sum
                # Skip the first column (date) and any empty or zero values
                log_product_sum = sum(
                    float(val) * math.log(float(val))
                    for val in row[1:]
                    if val.strip() and float(val) > 0
                )
                result.append(log_product_sum)
            except ValueError as e:
                logging.error(f"Error in row {row_num}: {e}")
                logging.info(f"Problematic row: {row}")
                # You can choose to either continue to the next row or raise the exception
                continue

    result_df = pd.DataFrame()
    df = pd.read_csv(file_path)
    result_df['date'] = df.iloc[:, 0]
    result_df['Theil'] = result

    result_df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    # result_df.set_index(df.iloc[:, 0], inplace=True)
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-poster')
    plt.title('Theil Index')
    plt.plot(result_df.iloc[:, 0], result_df['Theil'])
    plt.legend()
    plt.grid(visible=False)
    plt.show()

def make_quarterly(csv, differenced = True, dollarized = False):
    if differenced == False:
        df = pd.read_csv('data/original/' + csv)
    elif dollarized == True:
        df = pd.read_csv('data/dollarized/' + csv)
    else:
        df = pd.read_csv('data/differenced/' + csv)
    df = df.rename(columns={df.columns[0]: 'date'})
    df.set_index(df.columns[0], inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m')
    df_quarterly = df.resample('QE').sum()
    df_quarterly.index = pd.to_datetime(df_quarterly.index, format='%Y-%m')
    if dollarized == True:
        df_quarterly.to_csv('data/dollarized_quaterly/' + csv)
    else:
        df_quarterly.to_csv('data/quarterly/' + csv)

def dolarize(csv, differenced = True):
    usd = pd.read_csv('data/original/USD.csv')

    if differenced == False:
        df = pd.read_csv('data/original/' + csv)
    else:
        df = pd.read_csv('data/differenced/' + csv)
    df = df.rename(columns={df.columns[0]: 'date'})
    df_new = pd.DataFrame()

    for col in df.columns:
        if col not in ['date', '', 'Unnamed: 0']:
            df_new[col] = df[col] / usd['usd']
        # else:
        #     df_new[col] = df[col]
    df_new_new = pd.concat([df[df.columns[0]], df_new], axis=1)
    df_new_new.dropna()
    df_new_new.to_csv('data/dollarized/' + csv, index=False)

def transpose_resample(csv):
    df = pd.read_csv(csv)
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    df.reset_index(inplace=True)
    df = df.rename(columns={df.columns[0]: 'date'})
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d").strftime("%Y-%m")
    new_name = 'data/original/USD.csv'
    df.to_csv(new_name)
def get_loans_kved():
    start = '/Loans_KVED_'
    end = '-01.xlsx'
    for i in ['2020', '2021', '2022', '2023', '2024']:
        for j in range(1, 12):
            if i == '2024' and j == 7:
                break
            if j < 10:
                file_path = 'original_dataset/Loans_KVED/' + i + start + i + '-0' + str(j) + end
            else:
                file_path = 'original_dataset/Loans_KVED/' + i + start + i + '-' + str(j) + end
            if file_path not in ['original_dataset/Loans_KVED/2021/Loans_KVED_2021-01-01.xlsx', 'original_dataset/Loans_KVED/2020/Loans_KVED_2020-01-01.xlsx', 'original_dataset/Loans_KVED/2022/Loans_KVED_2022-04-01.xlsx', 'original_dataset/Loans_KVED/2022/Loans_KVED_2022-05-01.xlsx']:
                df = pd.read_excel(file_path, header=5)
                new_df = pd.DataFrame()
                bank_col = find_target_column(df, '2')
                kved_num_col = find_target_column(df, '3')
                kved_name_col = find_target_column(df, '4')
                credit_col = find_target_column(df, '5')
                npl_col = find_target_column(df, '8')
                if bank_col:
                    new_df['bank'] = df[bank_col]
                if kved_num_col:
                    new_df['kved_num'] = df[kved_num_col]
                if kved_name_col:
                    new_df['kved_name'] = df[kved_name_col]
                if credit_col:
                    new_df['credit'] = df[credit_col]
                if npl_col:
                    new_df['npl'] = df[npl_col]
                new_df.to_csv('original_dataset/loans_csv/' + i + '-' + str(j) + '.csv', index=False)

def process_loan_data(base_folder, bank_names_csv):
    # Read the bank names CSV
    bank_names_df = pd.read_csv(bank_names_csv, header=None, names=['English', 'Ukrainian'])
    bank_names = dict(zip(bank_names_df['Ukrainian'], bank_names_df['English']))

    # Dictionary to store data for each bank
    bank_data = {}

    # Iterate through years 2020 to 2024
    for year in range(2020, 2025):
        folder_path = os.path.join(base_folder, str(year))
        if not os.path.exists(folder_path):
            continue

        # Find the Excel file in the folder
        for file in os.listdir(folder_path):
            if file.startswith("Loans_KVED_") and file.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file)
                date_str = file[11:21]  # Extract date from filename
                date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # Read the Excel file
                df = pd.read_excel(file_path, usecols=[1, 2, 4, 7])
                df.columns = ['Bank', 'KVED', 'Loan_Amount', 'NPL_Amount']

                # Process each row in the dataframe
                for _, row in df.iterrows():
                    bank_ukr = str(row['Bank'])  # Convert to string to handle non-string values
                    if pd.isna(bank_ukr) or bank_ukr == '':
                        continue  # Skip rows with empty bank names
                    bank_eng = bank_names.get(bank_ukr, bank_ukr)  # Use English name if available, otherwise use Ukrainian
                    kved = row['KVED']
                    loan_amount = row['Loan_Amount']
                    npl_amount = row['NPL_Amount']

                    if bank_eng not in bank_data:
                        bank_data[bank_eng] = {'loans': {}, 'npl': {}}

                    if kved not in bank_data[bank_eng]['loans']:
                        bank_data[bank_eng]['loans'][kved] = {}
                        bank_data[bank_eng]['npl'][kved] = {}

                    bank_data[bank_eng]['loans'][kved][date] = loan_amount
                    bank_data[bank_eng]['npl'][kved][date] = npl_amount

    # Create CSV files for each bank
    for bank, data in bank_data.items():
        # Replace spaces with underscores in bank name and ensure it's a string
        bank_filename = str(bank).replace(' ', '_')

        # Create loans CSV
        loans_df = pd.DataFrame(data['loans'])
        loans_df.index.name = 'Date'
        loans_df.to_csv('data/loans/raw/loans/' + f"{bank_filename}_loans.csv")

        # Create NPL CSV
        npl_df = pd.DataFrame(data['npl'])
        npl_df.index.name = 'Date'
        npl_df.to_csv('data/loans/raw/npl/' + f"{bank_filename}_npl.csv")

    print("Processing complete. CSV files have been created for each bank.")


def merge_and_sort_csvs(csv_paths, name, npl):
    # Read and concatenate all CSV files
    df_list = [pd.read_csv('data/loans/raw/' + npl + '/' + trim_string(file, 21, 10) + '_' + npl + '.csv') for file in csv_paths]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Convert the first column to datetime
    merged_df.iloc[:, 0] = pd.to_datetime(merged_df.iloc[:, 0])

    # Sort by the first column (dates)
    sorted_df = merged_df.sort_values(by=merged_df.columns[0])

    # Reset the index
    sorted_df.reset_index(drop=True, inplace=True)

    sorted_df.to_csv('data/loans/grouped/' + npl + '/' + name + '.csv', index=False)

def delete_csv(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"The file {file_path} has been deleted successfully.")
        else:
            print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete {file_path}: {str(e)}")

def group_banks(list, name):
    merge_and_sort_csvs(list, name, 'npl')
    merge_and_sort_csvs(list, name, 'loans')
    # for bank in list:
        # delete_csv('data/loans/raw/npl/' + bank + '_npl.csv')
        # delete_csv('data/loans/raw/loans/' + bank + '_loans.csv')


def find_files_with_name(folder_path, name):
    matching_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if name.lower() in file.lower():
                matching_files.append(os.path.join(root, file))

    return matching_files

def trim_string(s, n, m):
    if len(s) < n + m:
        return ""
    return s[n:-m or None]

def group_banks_wrapper():
    bank_list = []

    with open('original_dataset/names.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row and row[0]:  # Check if the row is not empty
                bank = {
                    'english': row[0].strip(),
                    'ukrainian': row[1].strip() if len(row) > 1 else ''
                }
                bank_list.append(bank)

    # Print the result
    for bank in bank_list:
        print(bank['ukrainian'])
        group_banks(find_files_with_name('data/loans/raw/loans/', bank['ukrainian']), bank['english'])


