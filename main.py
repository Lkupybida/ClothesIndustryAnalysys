from data_extraction import *

if __name__=="__main__":
    # make_quarterly('total_income.csv')
    # make_quarterly('net_interest_income.csv')
    make_quarterly('total_assets.csv', True, True)
    # make_quarterly('administrative_expenses.csv')
    #make_quarterly('private_total_expenses.csv')
    #make_quarterly('private_equity_capital.csv', False)
    # make_quarterly('total_assets.csv', False)
    # make_quarterly('total_equity_capital.csv', False)
    # dolarize('total_assets.csv', False)