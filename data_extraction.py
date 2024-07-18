import os
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import openpyxl
import geopandas as gpd
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patheffects import withStroke
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple
import colorsys
import seaborn as sns
from matplotlib.sankey import Sankey
import holoviews as hv
import hvplot.pandas
from holoviews import opts
from matplotlib.ticker import FuncFormatter

hv.extension('bokeh')

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
    df_quarterly = df.resample('Q').sum()
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

def read_bank_filials(csv_file):
    # Read the CSV data
    data = pd.read_csv(csv_file, header=None)
    return data


def plot_bank_filials(bank):
    data = csv_for_geomapping('data/loans/regions.csv', bank)

    # Assuming the first row contains region names and the second row contains counts
    regions = data.iloc[0].tolist()
    counts = data.iloc[1].tolist()

    # Create a new DataFrame with regions and counts
    data = pd.DataFrame({'region': regions, 'count': counts})

    # Create a mapping between Ukrainian and English region names
    region_mapping = {
        'Вінніцька': 'Vinnytsya',
        'Волинська': 'Volyn',
        'Дніпропетровська': "Dnipropetrovs'k",
        'Донецька': "Donets'k",
        'Житомирська': 'Zhytomyr',
        'Закарпатська': 'Zakarpattia',
        'Запорізька': 'Zaporizhia',
        'ІваноФранківська': "Ivano-Frankivs'k",
        'Київ': 'Kiev City',
        'Київська': 'Kiev',
        'Кіровоградська': 'Kirovohrad',
        'Луганська': "Luhans'k",
        'Львівська': "L'viv",
        'Миколаївська': 'Mykolayiv',
        'Одеська': 'Odessa',
        'Полтавська': 'Poltava',
        'Рівненська': 'Rivne',
        'Сумська': 'Sumy',
        'Тернопільська': "Ternopil'",
        'Харківська': 'Kharkiv',
        'Херсонська': 'Kherson',
        'Хмельницька': "Khmel'nyts'kyy",
        'Черкаська': 'Cherkasy',
        'Чернігівська': 'Chernihiv',
        'Чернівецька': 'Chernivtsi',
        'АРК': 'Crimea'
    }

    # Map the Ukrainian names to English names
    data['region_en'] = data['region'].map(region_mapping)

    # Read the shapefile
    ukraine = gpd.read_file('original_dataset/gadm41_UKR_shp/gadm41_UKR_1.shp')

    # Merge the data with the shapefile
    ukraine = ukraine.merge(data, left_on='NAME_1', right_on='region_en', how='left')

    # Convert count to numeric, replacing NaN with 0
    ukraine['count'] = pd.to_numeric(ukraine['count'], errors='coerce').fillna(0)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    colors = ["#FFFFD8", "#D9EDBF", "#8ECAE6", "#003049"]
    n_bins = 10000  # Discretizes the interpolation into bins

    # Define custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_palette', colors, N=n_bins)

    # Plot the map with logarithmic scale
    norm = LogNorm(vmin=1, vmax=ukraine['count'].max())

    # Plot regions with count > 0
    regions_plot = ukraine[ukraine['count'] > 0].plot(column='count', ax=ax, legend=False,
                                                      cmap=custom_cmap, norm=norm)

    # Handle regions with count == 0 by setting them to light gray
    ukraine[ukraine['count'] == 0].plot(ax=ax, color='lightgrey', hatch='///')

    # Annotate each region with count value
    for idx, row in ukraine.iterrows():
        if row['count'] > 0:
            count_text = int(row['count'])  # Convert to integer
            text = str(count_text)

            # Lower the annotation for Kyivska oblast
            if row['region'] == 'Київська':
                ax.annotate(text, xy=(30.475674464445525, 50.29705971712133 - 0.2), color='white',
                            fontsize=14, ha='center', va='center', weight='bold',
                            path_effects=[withStroke(linewidth=1, foreground='black')])
            elif row['region'] == 'Одеська':
                ax.annotate(text, xy=(29.86530477280775 + 0.6, 46.744568162881855), color='white',
                            fontsize=14, ha='center', va='center', weight='bold',
                            path_effects=[withStroke(linewidth=1, foreground='black')])
            elif row['region'] == 'Київ':
                ax.annotate(text, xy=(30.522137666278727, 50.45778762272526 + 0.15), color='white',
                            fontsize=14, ha='center', va='center', weight='bold',
                            path_effects=[withStroke(linewidth=1, foreground='black')])
            else:
                ax.annotate(text, xy=row['geometry'].centroid.coords[0], color='white',
                            fontsize=14, ha='center', va='center', weight='bold',
                            path_effects=[withStroke(linewidth=1, foreground='black')])

    # Custom formatting function for the colorbar
    def format_ticks(value, tick_number):
        if value == 0:
            return '0'
        elif value < 1:
            return f'{value:.2f}'
        elif value == 2:
            return 'чайник'
        elif value == 3:
            return '3'
        elif value == 4:
            return '4'
        elif value == 5:
            return '5'
        else:
            return f'{int(value)}'

    # Update the colorbar with custom formatter
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm._A = []  # This is a workaround for a known bug in matplotlib
    print(sm)
    cbar = fig.colorbar(sm, ax=ax, format=FuncFormatter(format_ticks))

    # Remove axis and grid
    ax.axis('off')
    ax.grid(False)

    # Add a title
    plt.title('Кількість підрозділів ' + bank + 'у по регіонах (логарифмічний масштаб)', fontsize=16)

    # Show the plot
    plt.show()

def extract_filials():
    df = pd.read_excel('original_dataset/Kil_pidr_2024-07-01.xlsx', header=6, sheet_name='Діючі підрозділи_на 01.07.24')
    new_df = pd.DataFrame()
    bank_col = find_target_column(df, 'Назва банку')
    if bank_col:
        new_df['bank'] = df[bank_col]
    regions = ['Вінніцька','Волинська','Дніпропетровська','Донецька','Житомирська','Закарпатська','Запоріжська','ІваноФранківська','Київ','Київська','Кіровоградська','Луганська','Львівська','Миколаївська','Одеська','Полтавська','Рівненська','Сумська','Тернопільська','Харківська','Херсонська','Хмельницька','Черкаська','Чернігівська','Чернівецька','АРК']
    for r in regions:
        new_df[r] = df[r]

    new_df.to_csv('data/loans/regions.csv', index=False)


def process_csv_for_geomapping(file_path, entry):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if the entry is in the first column
    if entry not in df.iloc[:, 0].values:
        raise ValueError(f"Entry '{entry}' not found in the first column")

    # Find the row that matches the entry in the first column
    matched_row = df[df.iloc[:, 0] == entry].iloc[0]

    # Drop the first column
    df = df.drop(df.columns[0], axis=1)

    # Create a new DataFrame with the same column names
    new_df = pd.DataFrame(columns=df.columns)

    # Set the second row of the new DataFrame to the matched row
    new_df.loc[1] = matched_row[1:]

    return new_df


def csv_for_geomapping(csv_file, bank_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the row for the given bank
    bank_row = df[df['bank'] == bank_name]

    # Extract the column names (regions)
    columns = df.columns.tolist()[1:]  # Exclude the 'bank' column

    # Extract the values for the selected bank
    values = bank_row.values[0][1:].tolist()  # Exclude the 'bank' column

    # Create the new DataFrame in the desired format
    new_df = pd.DataFrame([columns, values])

    # Reset the index to have the desired format
    new_df.index = [0, 1]
    new_df.columns = list(range(len(columns)))

    return new_df

def read_shp(file_path):
    gdf = gpd.read_file(file_path)

    # Print the columns
    print(gdf.head())


def plot_top_5_columns(csv_path, dates):
    # Read the CSV file
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    # Set 'Date' column as index
    df.set_index('Date', inplace=True)

    # Create a figure with subplots for each date
    fig, axes = plt.subplots(len(dates), 1, figsize=(12, 6 * len(dates)), squeeze=False)
    fig.tight_layout(pad=5.0)

    for i, date in enumerate(dates):
        # Convert string to datetime if necessary
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        # Get the row for the specified date
        row = df.loc[date]

        # Sort the values and get the top 5
        top_5 = row.nlargest(5)

        # Create a horizontal bar plot for the top 5 values
        ax = axes[i, 0]
        top_5.plot(kind='barh', ax=ax)  # Changed kind to 'barh'

        # Customize the plot
        ax.set_title(f'Top 5 Columns on {date.strftime("%Y-%m-%d")}')
        ax.set_xlabel('Value')  # Adjusted x-axis label
        ax.set_ylabel('Column')  # Adjusted y-axis label
        ax.tick_params(axis='y', rotation=0)  # Rotate y-axis labels if needed

        # Add value labels on the right of each bar
        for j, v in enumerate(top_5):
            ax.text(v, j, f'{v:.2f}', ha='left', va='center')

    plt.show()


def plot_ranking_flow(csv_path, dates):
    # Read the CSV file
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    # Set 'Date' as index
    df.set_index('Date', inplace=True)

    # Get top 5 columns for each date
    top_5_by_date = {}
    for target_date in dates:
        # Find the closest date in the DataFrame
        closest_date = min(df.index, key=lambda x: abs(x - target_date))
        date_data = df.loc[closest_date]

        # Convert to numeric, replacing non-numeric values with NaN
        numeric_data = pd.to_numeric(date_data, errors='coerce')
        top_5 = numeric_data.nlargest(5)
        top_5_by_date[target_date] = top_5.index.tolist()

    # Prepare data for Sankey diagram
    nodes = []
    for date in dates:
        nodes.extend([f"{column} {date.strftime('%Y-%m')}" for column in top_5_by_date[date]])

    node_indices = {node: i for i, node in enumerate(nodes)}

    links_source = []
    links_target = []
    for i in range(len(dates) - 1):
        date1, date2 = dates[i], dates[i + 1]
        for column in top_5_by_date[date1]:
            if column in top_5_by_date[date2]:
                links_source.append(node_indices[f"{column} {date1.strftime('%Y-%m')}"])
                links_target.append(node_indices[f"{column} {date2.strftime('%Y-%m')}"])

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue"
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=[1] * len(links_source)
        ))])

    fig.update_layout(title_text="Top 5 Columns Ranking Flow", font_size=10)
    fig.show()


def create_sankey_diagram(csv_path, date_list):
    # Read the CSV file
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    # Filter the dataframe based on the provided dates
    df_filtered = df[df['Date'].isin(date_list)]

    # Sort the dataframe by date
    df_filtered = df_filtered.sort_values('Date')

    # Get the top 5 columns (excluding 'Date') based on the sum of values
    top_columns = df_filtered.drop('Date', axis=1).sum().nlargest(5).index.tolist()

    # Prepare data for Sankey diagram
    source = []
    target = []
    value = []
    label = []

    for i in range(len(df_filtered) - 1):
        current_date = df_filtered.iloc[i]['Date']
        next_date = df_filtered.iloc[i + 1]['Date']

        for col in top_columns:
            source.append(i * len(top_columns) + top_columns.index(col))
            target.append((i + 1) * len(top_columns) + top_columns.index(col))
            value.append(df_filtered.iloc[i + 1][col])

        label.extend([f"{col}\n{current_date.strftime('%Y-%m-%d')}" for col in top_columns])

    # Add labels for the last date
    label.extend([f"{col}\n{next_date.strftime('%Y-%m-%d')}" for col in top_columns])

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text="Top 5 Columns Flow Over Time", font_size=10)
    fig.show()

# Example usage:
# fig = create_sankey_diagram('path/to/your/csv/file.csv', ['2020-02-01', '2020-03-01', '2020-04-01'])
# fig.show()


def create_alluvial_diagram(csv_path: str, dates: List[str]) -> go.Figure:
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        print(f"CSV loaded. Shape: {df.shape}")

        # Filter the dataframe based on the specified dates
        df_filtered = df[df['Date'].isin(pd.to_datetime(dates))].sort_values('Date')
        print(f"Filtered dataframe shape: {df_filtered.shape}")

        if df_filtered.empty:
            print("No data found for the specified dates.")
            return go.Figure()

        # Get the top 5 columns (excluding 'Date') for each date
        top_columns = {}
        for date in df_filtered['Date']:
            date_df = df_filtered[df_filtered['Date'] == date].drop('Date', axis=1)
            top_5 = date_df.iloc[0].nlargest(5).index.tolist()
            top_columns[date] = top_5
        print(f"Top columns: {top_columns}")

        # Create a unique color for each unique column
        all_columns = list(set([col for cols in top_columns.values() for col in cols]))
        colors = [f'rgb{tuple(int(x * 255) for x in colorsys.hsv_to_rgb(i / len(all_columns), 0.8, 0.8))}'
                  for i in range(len(all_columns))]
        color_map = dict(zip(all_columns, colors))

        # Prepare data for the alluvial diagram
        node_labels = []
        node_colors = []
        source = []
        target = []
        link_colors = []

        for i, date in enumerate(df_filtered['Date']):
            current_top_5 = top_columns[date]
            node_labels.extend([f"{col} ({i + 1})" for i, col in enumerate(current_top_5)])
            node_colors.extend([color_map[col] for col in current_top_5])

            if i < len(df_filtered['Date']) - 1:
                next_date = df_filtered['Date'].iloc[i + 1]
                next_top_5 = top_columns[next_date]

                for j, col in enumerate(current_top_5):
                    if col in next_top_5:
                        source.append(i * 5 + j)
                        target.append((i + 1) * 5 + next_top_5.index(col))
                        link_colors.append(color_map[col])

        print(f"Node labels: {node_labels}")
        print(f"Source: {source}")
        print(f"Target: {target}")

        if not source or not target:
            print("No links found between dates.")
            return go.Figure()

        # Create the alluvial diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                color=link_colors
            ))])

        # Update layout
        date_labels = [date.strftime('%Y-%m-%d') for date in df_filtered['Date']]
        fig.update_layout(
            title_text="Alluvial Diagram of Top 5 Columns Over Time",
            font_size=10,
            annotations=[
                dict(
                    x=x,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=date,
                    showarrow=False,
                ) for x, date in zip([0, 0.5, 1], date_labels)
            ]
        )

        return fig

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return go.Figure()


def translate_columns(column_dict_path):
    # Load the dictionary CSV
    column_dict = pd.read_csv(column_dict_path, header=None, index_col=0)
    column_dict = column_dict.squeeze("columns").to_dict()
    return column_dict


def plot_alluvial_diagram(data_csv_path, column_dict_path, dates):
    # Load the column dictionary
    column_dict = translate_columns(column_dict_path)

    # Load the data CSV
    df = pd.read_csv(data_csv_path, parse_dates=['Date'])

    # Translate column names
    translated_columns = {col: column_dict[col] for col in df.columns if col in column_dict}
    df = df.rename(columns=translated_columns)

    # Filter the dataframe to only include the specified dates
    df_filtered = df[df['Date'].isin(pd.to_datetime(dates))]

    # Prepare the data for the alluvial diagram
    alluvial_data = []
    for date in dates:
        date_data = df_filtered[df_filtered['Date'] == date].iloc[:, 1:].T
        date_data.columns = ['value']
        date_data['column'] = date_data.index
        date_data['date'] = date
        date_data = date_data.sort_values(by='value', ascending=False).head(5)
        alluvial_data.append(date_data)

    alluvial_data = pd.concat(alluvial_data)

    # Map columns to categories for visualization
    alluvial_data['column'] = alluvial_data['column'].astype('category')

    # Define the colors for the top 5 columns
    colors = ["#FFFFD8", "#D9EDBF", "#8ECAE6", "#003049"]

    # Create a color mapping based on rank
    color_mapping = {rank: color for rank, color in enumerate(colors)}

    # Plot the alluvial diagram
    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")

    # Use a scatter plot to simulate the alluvial diagram
    for i in range(len(dates) - 1):
        left = alluvial_data[alluvial_data['date'] == dates[i]].reset_index(drop=True)
        right = alluvial_data[alluvial_data['date'] == dates[i + 1]].reset_index(drop=True)

        # Sort by value to ensure largest number is on top
        left = left.sort_values(by='value', ascending=False)
        right = right.sort_values(by='value', ascending=False)

        for j in range(len(left)):
            rank = j
            color = color_mapping.get(rank, "#003049")
            plt.plot([i, i + 1], [left['column'].cat.codes[j], right['column'].cat.codes[j]], color=color, lw=3)

    # Create a legend for the colors
    legend_labels = [f"Rank {rank+1}" for rank in range(len(colors))]
    plt.legend(legend_labels, loc='upper right')

    plt.xticks(range(len(dates)), dates)
    plt.xlabel('Date')
    plt.ylabel('Top 5 Columns')
    plt.title('Alluvial Diagram of Top 5 Columns Over Time')
    plt.grid(False)

    # Remove text on the plot
    plt.texts = []

    plt.show()



def read_unique_csv(file_path, out):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Drop duplicate rows
    df_unique = df.drop_duplicates()

    # Convert the DataFrame back to a list of lists
    unique_rows = df_unique.values.tolist()

    df_unique.to_csv(out, index=False, header=False)


def plot_alluvial_diagram2(main_csv_path, dict_csv_path, dates):
    # Read the main data CSV file
    main_df = pd.read_csv(main_csv_path, parse_dates=['Date'])

    # Filter data for the specified dates
    filtered_df = main_df[main_df['Date'].isin(pd.to_datetime(dates))]

    # Read the dictionary CSV file
    dict_df = pd.read_csv(dict_csv_path, header=None, index_col=0)
    dict_map = dict_df[1].to_dict()

    # Translate column names
    translated_columns = ['Date'] + [dict_map.get(col, col) for col in main_df.columns[1:]]
    filtered_df.columns = translated_columns

    # Extract the top 5 columns for each date
    top_columns_per_date = {}
    for date in dates:
        date_data = filtered_df[filtered_df['Date'] == date].drop(columns='Date').squeeze()
        top_columns = date_data.nlargest(5).index.tolist()
        top_columns_per_date[date] = top_columns

    # Prepare data for the alluvial diagram
    alluvial_data = []
    for i in range(len(dates) - 1):
        start_date = dates[i]
        end_date = dates[i + 1]

        start_top = top_columns_per_date[start_date]
        end_top = top_columns_per_date[end_date]

        for start_col in start_top:
            if start_col in end_top:
                alluvial_data.append((start_date, start_col, end_date, start_col, 1))
            else:
                for end_col in end_top:
                    alluvial_data.append((start_date, start_col, end_date, end_col, 0.2))

    alluvial_df = pd.DataFrame(alluvial_data, columns=['StartDate', 'StartCategory', 'EndDate', 'EndCategory', 'Value'])

    # Plot the alluvial diagram
    alluvial_plot = alluvial_df.hvplot.sankey(
        'StartCategory', 'EndCategory', 'Value', color='StartCategory',
        labels='Value', title='Alluvial Diagram'
    ).opts(
        opts.Sankey(width=800, height=600, edge_color='StartCategory', node_color='StartCategory',
                    label_position='outer')
    )

    hv.save(alluvial_plot, 'alluvial_diagram.html', backend='bokeh')
    return alluvial_plot


def create_alluvial_diagram_2(csv_path, column_dict_path, dates):
    # Read the CSV file
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    # Read the column dictionary
    column_dict = pd.read_csv(column_dict_path, header=None, index_col=0).iloc[:, 0].to_dict()
    # Filter the dataframe for the specified dates
    df_filtered = df[df['Date'].isin(dates)]

    # Get the top 5 columns for each date
    top_columns = []
    for date in dates:
        date_data = df_filtered[df_filtered['Date'] == date].iloc[0]
        numeric_columns = date_data.drop('Date')
        numeric_columns = numeric_columns[numeric_columns.apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull()]
        top_5 = numeric_columns.nlargest(5).index.tolist()
        top_columns.extend(top_5)

    top_columns = list(set(top_columns))  # Remove duplicates

    # Prepare data for the alluvial diagram
    node_labels = []
    source = []
    target = []
    value = []

    for i, date in enumerate(dates):
        date_data = df_filtered[df_filtered['Date'] == date].iloc[0]

        for col in top_columns:
            node_labels.append(f"{column_dict.get(col, col)} ({date.strftime('%Y-%m-%d')})")

            if i < len(dates) - 1:
                source.append(len(node_labels) - 1)
                target.append(len(node_labels) - 1 + len(top_columns))
                value.append(1)  # Constant width

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text="Alluvial Diagram of Top 5 Columns Over Time", font_size=10)
    fig.show()


def create_alluvial_diagram_3(data_path, dict_path, dates):
    # Read the data and dictionary
    df = pd.read_csv(data_path, parse_dates=['Date'])
    dict_df = pd.read_csv(dict_path, header=None, index_col=0)

    # Translate column names
    column_names = {col: dict_df.loc[col, 1] if col in dict_df.index else col for col in df.columns}
    df.rename(columns=column_names, inplace=True)

    # Filter data for specified dates
    df_filtered = df[df['Date'].isin(dates)]

    # Identify numeric columns
    numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

    # Get top 5 columns for each date
    top_columns = []
    for date in dates:
        date_data = df_filtered[df_filtered['Date'] == date].iloc[0]
        top_5 = date_data[numeric_columns].nlargest(5).index.tolist()
        top_columns.extend(top_5)

    top_columns = list(set(top_columns))

    # Prepare data for alluvial diagram
    alluvial_data = []
    for date in dates:
        date_data = df_filtered[df_filtered['Date'] == date].iloc[0]
        sorted_cols = date_data[top_columns].sort_values(ascending=False)
        for i, (col, value) in enumerate(sorted_cols.items()):
            alluvial_data.append({'Date': date, 'Column': col, 'Rank': i + 1, 'Value': value})

    alluvial_df = pd.DataFrame(alluvial_data)

    # Create alluvial diagram
    plt.figure(figsize=(12, 8))

    for col in top_columns:
        col_data = alluvial_df[alluvial_df['Column'] == col]
        plt.plot(col_data['Date'], col_data['Rank'], '-o', linewidth=2, markersize=8)

        for i in range(len(col_data) - 1):
            x1, y1 = col_data.iloc[i]['Date'], col_data.iloc[i]['Rank']
            x2, y2 = col_data.iloc[i + 1]['Date'], col_data.iloc[i + 1]['Rank']
            plt.fill_between([x1, x2], [y1, y2], [y1 + 0.8, y2 + 0.8], alpha=0.3)

    plt.gca().invert_yaxis()
    plt.ylabel('Rank')
    plt.title('Alluvial Diagram of Top 5 Columns')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation

    for i, date in enumerate(dates):
        date_data = alluvial_df[alluvial_df['Date'] == date]
        for _, row in date_data.iterrows():
            plt.text(row['Date'], row['Rank'], row['Column'], ha='right' if i == 0 else 'left', va='center')

    plt.tight_layout()
    plt.show()

# Define your dictionary mapping codes to descriptions
code_to_description = {
    '00': 'Інше (для фізичних осіб (у т. ч. суб`єктів незалежної професійної діяльності) та нерезидентів)',
    '01': 'Сільське господарство, мисливство та надання пов\'язаних із ними послуг',
    '02': 'Лісове господарство та лісозаготівлі',
    '03': 'Рибне господарство',
    '05': 'Добування кам\'яного та бурого вугілля',
    '06': 'Добування сирої нафти та природного газу',
    '07': 'Добування металевих руд',
    '08': 'Добування інших корисних копалин та розроблення кар\'єрів',
    '09': 'Надання допоміжних послуг у сфері добувної промисловості та розроблення кар\'єрів',
    '10': 'Виробництво харчових продуктів',
    '11': 'Виробництво напоїв',
    '12': 'Виробництво тютюнових виробів',
    '13': 'Текстильне виробництво',
    '14': 'Виробництво одягу',
    '15': 'Виробництво шкіри, виробів зі шкіри та інших матеріалів',
    '16': 'Оброблення деревини та виготовлення виробів з деревини та корка, крім меблів; виготовлення виробів із соломки та рослинних матеріалів для плетіння',
    '17': 'Виробництво паперу та паперових виробів',
    '18': 'Поліграфічна діяльність, тиражування записаної інформації',
    '19': 'Виробництво коксу та продуктів нафтоперероблення',
    '20': 'Виробництво хімічних речовин і хімічної продукції',
    '21': 'Виробництво основних фармацевтичних продуктів і фармацевтичних препаратів',
    '22': 'Виробництво гумових і пластмасових виробів',
    '23': 'Виробництво іншої неметалевої мінеральної продукції',
    '24': 'Металургійне виробництво',
    '25': 'Виробництво готових металевих виробів, крім машин і устатковання',
    '26': 'Виробництво комп\'ютерів, електронної та оптичної продукції',
    '27': 'Виробництво електричного устатковання',
    '28': 'Виробництво машин і устатковання, н.в.і.у.',
    '29': 'Виробництво автотранспортних засобів, причепів і напівпричепів',
    '30': 'Виробництво інших транспортних засобів',
    '31': 'Виробництво меблів',
    '32': 'Виробництво іншої продукції',
    '33': 'Ремонт і монтаж машин і устатковання',
    '35': 'Постачання електроенергії, газу, пари та кондиційованого повітря',
    '36': 'Забір, очищення та постачання води',
    '37': 'Каналізація, відведення й очищення стічних вод',
    '38': 'Збирання, оброблення й видалення відходів; відновлення матеріалів',
    '39': 'Інша діяльність щодо поводження з відходами',
    '41': 'Будівництво будівель',
    '42': 'Будівництво споруд',
    '43': 'Спеціалізовані будівельні роботи',
    '45': 'Оптова та роздрібна торгівля автотранспортними засобами та мотоциклами, їх ремонт',
    '46': 'Оптова торгівля, крім торгівлі автотранспортними засобами та мотоциклами',
    '47': 'Роздрібна торгівля, крім торгівлі автотранспортними засобами та мотоциклами',
    '49': 'Наземний і трубопровідний транспорт',
    '50': 'Водний транспорт',
    '51': 'Авіаційний транспорт',
    '52': 'Складське господарство та допоміжна діяльність у сфері транспорту',
    '53': 'Поштова та кур\'єрська діяльність',
    '55': 'Тимчасове розміщування',
    '56': 'Діяльність із забезпечення стравами та напоями',
    '58': 'Видавнича діяльність',
    '59': 'Виробництво кіно- та відеофільмів, телевізійних програм, видання звукозаписів',
    '60': 'Діяльність у сфері радіомовлення та телевізійного мовлення',
    '61': 'Телекомунікації (електрозв\'язок)',
    '62': 'Комп\'ютерне програмування, консультування та пов\'язана з ними діяльність',
    '63': 'Надання інформаційних послуг',
    '64': "Надання фінансових послуг, крім страхування та пенсійного забезпечення",
    '65': "Страхування, перестрахування та недержавне пенсійне забезпечення, крім обов'язкового соціального страхування",
    '66': "Допоміжна діяльність у сферах фінансових послуг і страхування",
    '68': "Операції з нерухомим майном",
    '69': "Діяльність у сферах права та бухгалтерського обліку",
    '70': "Діяльність головних управлінь (хед-офісів); консультування з питань керування",
    '71': "Діяльність у сферах архітектури та інжинірингу; технічні випробування та дослідження",
    '72': "Наукові дослідження та розробки",
    '73': "Рекламна діяльність і дослідження кон'юнктури ринку",
    '74': "Інша професійна, наукова та технічна діяльність",
    '75': "Ветеринарна діяльність",
    '77': "Оренда, прокат і лізинг",
    '78': "Діяльність із працевлаштування",
    '79': "Діяльність туристичних агентств, туристичних операторів, надання інших послуг із бронювання та пов'язана з цим діяльність",
    '80': "Діяльність охоронних служб та проведення розслідувань",
    '81': "Обслуговування будинків і територій",
    '82': "Адміністративна та допоміжна офісна діяльність, інші допоміжні комерційні послуги",
    '84': "Державне управління й оборона; обов'язкове соціальне страхування",
    '85': "Освіта",
    '86': "Охорона здоров'я",
    '87': "Надання послуг догляду із забезпеченням проживання",
    '88': "Надання соціальної допомоги без забезпечення проживання",
    '90': "Діяльність у сфері творчості, мистецтва та розваг",
    '91': "Функціювання бібліотек, архівів, музеїв та інших закладів культури",
    '92': "Організування азартних ігор",
    '93': "Діяльність у сфері спорту, організування відпочинку та розваг",
    '94': "Діяльність громадських організацій",
    '95': "Ремонт комп'ютерів, побутових виробів і предметів особистого вжитку",
    '96': "Надання інших індивідуальних послуг",
    '97': "Діяльність домашніх господарств як роботодавців для домашньої прислуги"
}

def rename_columns_in_csv(input_file, output_file):
    # Read the CSV file
    if input_file not in ['data/loans/grouped/loans/.csv', 'data/loans/grouped/loans/nan.csv']:
        df = pd.read_csv(input_file)

    # Rename columns using the dictionary
        new_columns = [code_to_description.get(col.strip(), col.strip()) for col in df.columns]
        df.columns = new_columns

    # Save the updated CSV file
        df.to_csv(output_file, index=False)

def make_yearly(path_in, csv, path_out):
    df = pd.read_csv(path_in + csv)

    df = df.rename(columns={df.columns[0]: 'date'})
    df.set_index(df.columns[0], inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df_quarterly = df.resample('Y').sum()
    df_quarterly.index = pd.to_datetime(df_quarterly.index, format='%Y-%m')
    df_quarterly.to_csv(path_out + csv)


