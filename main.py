from data_extraction import *

if __name__=="__main__":
    make_quarterly('total_income.csv')
    make_quarterly('net_interest_income.csv')
    make_quarterly('total_expenses.csv')
    make_quarterly('administrative_expenses.csv')
    make_quarterly('total_assets.csv', False)
    make_quarterly('total_equity_capital.csv', False)