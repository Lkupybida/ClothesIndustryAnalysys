import os
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import DateOffset

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
    values.append(3433067383.21959)

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

def make_quarterly(csv, differenced =  True):
    if differenced == False:
        df = pd.read_csv('data/original/' + csv)
    else:
        df = pd.read_csv('data/differenced/' + csv)
    df = df.rename(columns={df.columns[0]: 'date'})
    df.set_index(df.columns[0], inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m')
    df_quarterly = df.resample('QE').sum()
    df_quarterly.index = pd.to_datetime(df_quarterly.index, format='%Y-%m')
    df_quarterly.to_csv('data/quarterly/' + csv)