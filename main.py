from data_extraction import *

if __name__=="__main__":
    # make_quarterly('total_income.csv')
    # make_quarterly('net_interest_income.csv')
    # make_quarterly('administrative_expenses.csv')
    #make_quarterly('private_total_expenses.csv')
    #make_quarterly('private_equity_capital.csv', False)
    # make_quarterly('total_assets.csv', False)
    # make_quarterly('total_equity_capital.csv', False)
    # dolarize('TA.csv', False)
    make_quarterly('private_deposit.csv', False)
    # get_loans_kved()

    # process_loan_data('original_dataset/Loans_KVED', 'original_dataset/names.csv')
    group_banks_wrapper()