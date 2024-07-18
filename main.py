from data_extraction import *

if __name__=="__main__":
    # make_quarterly('total_income.csv')
    # make_quarterly('net_interest_income.csv')
    # make_quarterly('administrative_expenses.csv')
    # make_quarterly('private_total_expenses.csv')
    # make_quarterly('private_equity_capital.csv', False)
    # make_quarterly('total_assets.csv', False)
    # make_quarterly('total_equity_capital.csv', False)
    # dolarize('total_income.csv')
    # dolarize('total_expenses.csv')
    make_quarterly('total_assets_all.csv', False, False)
    # make_quarterly('total_expenses.csv', True, True)
    # make_quarterly('iglb.csv', False)
    # get_loans_kved()

    # process_loan_data('original_dataset/Loans_KVED', 'original_dataset/names.csv')
    # group_banks_wrapper()

    #plot_bank_filials('original_dataset/filials_oschad.csv')
    #extract_filials()
    # read_unique_csv('original_dataset/Loans_KVED_2024-06-01 - Sheet1.csv', 'original_dataset/Loans_KVED_2024-06-01 - Sheet2.csv')
    # bank_names_df = pd.read_csv('original_dataset/names.csv', header=None, names=['English', 'Ukrainian'])
    # for bank in bank_names_df['English']:
    #     rename_columns_in_csv('data/loans/grouped/loans/' + str(bank) + '.csv', 'data/loans/kved_named/loans/' + str(bank) + '.csv')