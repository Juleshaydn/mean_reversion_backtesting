import pandas as pd

def calculate_allocation(portfolio):
    total_value = portfolio['Current Value'].sum()
    portfolio['Allocation'] = portfolio['Current Value'] / total_value
    return portfolio, total_value

def rebalance_portfolio(portfolio, total_value, target_allocation):
    portfolio['Target Value'] = portfolio['Asset'].map(target_allocation) * total_value
    portfolio['Difference'] = portfolio['Target Value'] - portfolio['Current Value']
    return portfolio
