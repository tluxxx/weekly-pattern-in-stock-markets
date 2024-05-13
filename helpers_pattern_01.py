# helper functions code for weekly pattern programs
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tabulate import tabulate
import numba as nb
from numba import jit


# helper functions for resampling daily to weekly data
def transforming_daily_weekly(pr, mode):
  ''' resampling daily to weekly data, numbering weeks, 
      mode = 'close' original strategy --> trading first day of the week at Close
      mode = 'open' modified strategy --> trading first day of the week at Open 
  '''
  pr['date'] = pr.index
  if mode=='close':
      df_input = {'start_w': pr.date.resample('W').first(),
      'price': pr.Close.resample('W').first()}
  elif mode=='open':
      df_input = {'start_w': pr.date.resample('W').first(),
      'price': pr.Open.resample('W').first()}
  else:
      print(' error')
      return
  pr_w = pd.DataFrame(df_input)
  pr_w = pr_w.assign(week_nb=range(len(pr_w)))
  pr_w.set_index('start_w', inplace=True)
  return pr_w

def pos_weekly(pr_w, pos, shift):
  ''' matching week-numbers with with weekly pattern in a cycle
  '''
  pos_dict = dict(enumerate(pos))
  # calculation of week-type and resulting positions depending on the shift
  week_type = (pr_w['week_nb'] + shift) % len(pos)
  return week_type.map(pos_dict)

# detection of patterns in weekly returns
def pattern_detection(pr_w, n, upper_limit, lower_limit):
  ''' calculation of retrurns per week_type and defining pattern using return thresholds
  '''
  df1 = pd.DataFrame({'week_type': (pr_w['week_nb']) % n,
                      'returns': pr_w['price'].pct_change().shift(-1)})
  df1 = (df1.groupby(['week_type'])['returns'].mean() * 100)
  ddf1 = df1.to_numpy()
  pos = np.where(ddf1 > upper_limit, 1, np.where(ddf1 < lower_limit, -1, 0)).tolist()
  return df1, pos


# Helper functions for PnL calculations
def pnl_accumulation(pr_w, px):
  ''' calculation of PnL-curve for a given weekly pattern, accumulation (i.e. investing all returns)
  '''
  rets = pr_w['price'].pct_change() * px.shift(1)
  return (1 + rets).cumprod()

def pnl_constant_invest(pr_w, px):
  ''' calculation of PnL-curve for a given weekly pattern, accumulation (i.e. investing all returns)
  '''
  rets = pr_w['price'].pct_change() * px.shift(1)
  return 1 + rets.cumsum()   


@jit (nopython=True)
def pnl_calc (eq_w_fee, rets, pos_change, variable_fee, fixed_fee, fee, rows):
    for row in range(1, rows):
        eq_w_fee[row] = (1 + rets[row]) * eq_w_fee[row-1]
        variable_fee[row] = eq_w_fee[row] * pos_change[row] * fee
        eq_w_fee[row] = eq_w_fee[row] - variable_fee[row] - fixed_fee[row]

def pnl_acc_real_equity(pr_w, px, start_equity, fee, fixed_fee):
  ''' real equity curve incl. fee
  '''
  ## vectorized part: calculation for input to equity-curves with fee 
  rets = (pr_w['price'].pct_change() * px.shift(1)).to_numpy()
  pos_change = abs(px.diff()).to_numpy()
  fixed_fee = (abs(px.diff()) * fixed_fee).to_numpy()
  rows = len(px)
  variable_fee = np.zeros(rows)
  eq_w_fee = np.zeros(rows)
  eq_w_fee[0] = start_equity
  ## looping part, using numpy to accelerate the loopign
  pnl_calc(eq_w_fee,rets, pos_change, variable_fee, fixed_fee, fee, rows) 
  return eq_w_fee
  

# helper function for plotting
def position_gantt(tls, labels, head_line):
  ''' function to produce gantt-charts of various trade-timelines
  '''
  gantt = []
  nb_rows = len(labels)
  for count, tl in enumerate(tls):
    task = labels[count]
    for i in range (len(tl)):
      start = tl.loc[i, 'date_entry']
      end = tl.loc[i, 'date_exit']
      typ = tl.loc[i, 'type']
      x = dict(Task=task, Start=start, Finish=end, Resource=typ)
      gantt.append(x)
  fig = px.timeline(gantt, x_start="Start", x_end="Finish", y="Task", color="Resource", title=head_line)
  fig.update_layout(xaxis_title='Date',yaxis_title='strategy')
  fig.update_layout(template = 'plotly_dark', autosize=False, width=1600, height=nb_rows*80)
  fig.show()
  

# helper functions for trade statistics
def calculation_entry_exits(pr_w, px):
  ''' re-calculation of entries and exits from positions
  '''
  tr = pd.DataFrame()
  tr['price'] = pr_w['price']
  tr['long_entry'] = np.where(((px ==1) & (px.shift(1) !=1)), 1, np.nan)
  tr['long_exit'] = np.where(((px !=1) & (px.shift(1) ==1)), 1, np.nan)
  tr['short_entry'] = np.where(((px ==-1) & (px.shift(1) !=-1)), 1, np.nan)
  tr['short_exit'] = np.where(((px !=-1) & (px.shift(1) ==-1)), 1, np.nan)
  tr['date'] = tr.index
  return tr

def trading_journal(pr_w, px):
  ''' tabulating from positions the list of trades with some basic characteristics
  '''
  # adding entries and exits
  trd_help = calculation_entry_exits(pr_w, px)
  # long trades
  buffer1 = trd_help[trd_help['long_entry']==1][['date','price']]
  buffer2 = trd_help[trd_help['long_exit']==1][['date','price']]
  buffer1.columns = ['date_entry', 'price_entry']
  buffer2.columns = ['date_exit', 'price_exit']
  buffer1['ind'] = list(range(1, len(buffer1) + 1))
  buffer2['ind'] = list(range(1, len(buffer2) + 1))
  buffer1.set_index('ind', inplace=True)
  buffer2.set_index('ind', inplace=True)
  buffer1['type'] = 'LONG'
  buffer1['trade_id'] =[f'TR-L-{i+1}' for i in buffer1.index]
  long_trades = buffer1.merge(buffer2, how="outer", left_index=True, right_index=True)
  # short trades
  buffer1 = trd_help[trd_help['short_entry']==1][['date','price']]
  buffer2 = trd_help[trd_help['short_exit']==1][['date','price']]
  buffer1.columns = ['date_entry', 'price_entry']
  buffer2.columns = ['date_exit', 'price_exit']
  buffer1['ind'] = list(range(1, len(buffer1) + 1))
  buffer2['ind'] = list(range(1, len(buffer2) + 1))
  buffer1.set_index('ind', inplace=True)
  buffer2.set_index('ind', inplace=True)
  buffer1['type'] = 'SHORT'
  buffer1['trade_id'] =[f'TR-S-{i+1}' for i in buffer1.index]
  short_trades = buffer1.merge(buffer2, how="outer", left_index=True, right_index=True)
  # combining and basic analyses
  trades = pd.concat([long_trades, short_trades], ignore_index=True, sort=False).sort_values(by=['date_entry'])
  rets = (trades['price_exit'] - trades['price_entry']) / trades['price_entry']
  trades['returns'] = np.where(trades['type']=='LONG', rets, -rets)
  trades['duration'] = trades['date_exit'] -trades['date_entry']
  return trades


def trade_key_parameters(trd, pr_w):
  ''' calculating and printing key statistics (independed from start equity)
  '''
  long_trades = trd[trd['type'] == 'LONG']
  short_trades = trd[trd['type'] == 'SHORT']
  nb_trades = len(trd)
  nb_long_trades = len(long_trades)
  nb_short_trades = nb_trades - nb_long_trades
  win_ratio = len(trd[trd['returns'] > 0])/nb_trades *100
  win_ratio_long = len(long_trades[long_trades['returns'] > 0]) / nb_long_trades * 100
  win_ratio_short = len(short_trades[short_trades['returns'] > 0]) / nb_short_trades * 100
  winning_trades = (trd[trd['returns'] > 0])
  loosing_trades = (trd[trd['returns'] < 0])
  profit_factor = - winning_trades['returns'].sum()/loosing_trades['returns'].sum()
  average_trade_profit = trd['returns'].mean()*100
  average_trade_profit_long = trd[trd['type']=='LONG']['returns'].mean()*100
  average_trade_profit_short = trd[trd['type']=='SHORT']['returns'].mean()*100
  largest_looser = trd['returns'].min()*100
  largest_looser_long = long_trades['returns'].min() * 100
  largest_looser_short = short_trades['returns'].min() *100
  
  start_date, end_date = pr_w.index[0], pr_w.index[len(pr_w)-1]
  total_duration = end_date - start_date
  duration_long = long_trades['duration'].sum() / total_duration * 100
  duration_short = short_trades['duration'].sum() / total_duration * 100
  duration_flat = 100 - duration_long - duration_short
  
  # print aout section (using tabulate) 
  print(f'********************************************************')
  print(f'start date:  {start_date.date()}')
  print(f'end date:    {end_date.date()}')
  headers = [' Parameter', 'All Trades', 'LONG',  'SHORT',  'FLAT']
  t01 = ['duration (days): ', total_duration.days]
  t02 = ['nb of trades: ', nb_trades, nb_long_trades, nb_short_trades]
  t03 = ['duration of trades (%): ', 100, duration_long, duration_short, duration_flat]
  t04 = ['percent profitable (%): ', win_ratio, win_ratio_long, win_ratio_short]
  t05 = ['profit factor: ', profit_factor]
  t06 = ['average trade profit (%)', average_trade_profit, average_trade_profit_long, average_trade_profit_short]
  t07 = ['largest loosing trade (%)', largest_looser, largest_looser_long, largest_looser_short]
  table = [t01, t02, t03, t04, t05, t06, t07]
  print(tabulate(table, headers, tablefmt='simple_grid', intfmt=',', floatfmt=',.2f'))
   

def trade_key_parameters2(trades, start_equity, fee, fixed_fee):
  ''' calculating and printing key statistics (depended from start equity)
  '''
    # adding new columns
  trades['equity_wo_fee'] = 0
  trades['equity'] = 0
  trades['fee_open'] = 0
  trades['fee_exit'] = 0
  # inital values for first trade
  trades.loc[0,'equity_wo_fee'] = start_equity * (1 + trades.loc[0, 'returns'])
  trades.loc[0,'fee_open'] = start_equity * fee + fixed_fee
  equity = (start_equity - trades.loc[0, 'fee_open']) * (1 + trades.loc[0, 'returns'])
  exit_fee = equity * fee + fixed_fee
  trades.loc[0, 'fee_exit'] = exit_fee
  trades.loc[0, 'equity'] = equity - exit_fee

  # iteration over the other trades
  for row in range(1, len(trades)):
    r, r1 = trades.index[row], trades.index[row-1]
    trades.loc[r,'equity_wo_fee'] = trades.loc[r1,'equity_wo_fee'] * (1 + trades.loc[r,'returns'])
    # copening a position considering fee
    trades.loc[r,'fee_open'] = trades.loc[r1, 'equity'] * fee + fixed_fee
    equity = (trades.loc[r1, 'equity'] - trades.loc[r,'fee_open']) * (1 + trades.loc[r,'returns'])
    exit_fee = equity * fee + fixed_fee
    trades.loc[r, 'fee_exit'] = exit_fee
    trades.loc[r, 'equity'] = equity - exit_fee
  # coorrection of last trade is still open
  trades['equity_wo_fee'].fillna(method='ffill', inplace=True)
  trades['equity'].fillna(method='ffill', inplace=True)

  # summary
  r = trades.index[len(trades)-1],
  final_equity_wo_fee = trades.loc[r, 'equity_wo_fee']
  final_equity = trades.loc[r, 'equity']
  total_fees = trades.fee_open.sum() + trades.fee_exit.sum()
  gross_profit = final_equity_wo_fee - start_equity
  gross_profit_p = gross_profit / start_equity * 100
  net_profit = final_equity - start_equity
  net_profit_p = net_profit / start_equity * 100
  
  # printing section using tabulate 
  print(f'********************************************************')
  headers = [' Parameter', 'value', '(%)']
  t01 = ['start equity:', start_equity]
  t02 = ['final equity (w/o fee): ', final_equity_wo_fee]
  t03 = ['final equity (incl. fee): ', final_equity]
  t04 = ['gross_profit (w/o fee): ', gross_profit, gross_profit_p]
  t05 = ['net_profit (incl.fee):', net_profit, net_profit_p]
  t06 = ['total fees: ', total_fees]
  table = [t01, t02, t03, t04, t05, t06]
  print(tabulate(table, headers, tablefmt='simple_grid', intfmt=',', floatfmt=',.2f')) 
  
  
