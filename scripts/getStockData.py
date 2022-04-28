#%% import yahoo finance api
import yfinance as yf
import datetime
import pandas as pd

#%% Set start and end times
# startDate , as per our convenience we can modify
startDate = datetime.datetime(1996, 6, 15)
 
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2002, 12, 15)

# Get stock tickers
tickersDJIA1999 = ["AA", "XOM","MCD","HON","GE",
                    "MRK","AXP","MSFT","T",
                    "MMM","BA","HD",
                    "CAT","INTC","PG","TRV","IBM",
                    "KO","IP","RTX","DD","JNJ",
                    "WMT","JPM","DIS",
                    "HP","MO"] #SBC communications merged with AT&T and trades as T now...
                    # GM and KODK has no daily stock price before 2009 and 2013 because they went bankrupt and the old stock data is not publically published

#%%
for name in tickersDJIA1999:
    tickerInfo = yf.Ticker(name)
    f = open("/home/nikap/Desktop/Masterthesis/Conic_Portfolio/data/"+name+"_2000_stock.csv", "a")
    f.write(pd.DataFrame.to_csv(tickerInfo.history(start=startDate,end=endDate)))
    f.close()
#%% Set start and end times 2008
# startDate , as per our convenience we can modify
startDate = datetime.datetime(2005, 2, 15)
 
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2009, 3, 15)

# Get stock tickers
tickersDJIA2005 = ["MMM","DD","JPM",
                    "AA", "XOM","MCD",
                    "MO","GE","MRK",
                    "AXP"   ,"MSFT",
                    "AIG","HP","PFE",
                    "T","HD","PG",
                    "BA","HON","RTX",
                    "CAT","INTC","VZ",
                    "C","IBM","WMT",
                    "KO","JNJ","DIS"] 
                    # GM sold new shares around 2009 and the old share prices are not published online

#%%
for name in tickersDJIA2005:
    tickerInfo = yf.Ticker(name)
    f = open("/home/nikap/Desktop/Masterthesis/Conic_Portfolio/data/DJ2008/"+name+"_2008_stock.csv", "a")
    f.write(pd.DataFrame.to_csv(tickerInfo.history(start=startDate,end=endDate)))
    f.close()

#%%
tickerInfo = yf.Ticker("^DJI")
f = open("/home/nikap/Desktop/Masterthesis/Conic_Portfolio/data/DJ2020/DJI_2020_stock.csv", "a")
f.write(pd.DataFrame.to_csv(tickerInfo.history(start=startDate,end=endDate)))
f.close()

#%% Set start and end times 2008
# startDate , as per our convenience we can modify
startDate = datetime.datetime(2012, 5, 15)
 
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2016, 2, 15)

# Get stock tickers
tickersDJIA2016 = ["MMM","GE","NKE",
                    "AXP","GS","PFE",
                    "AAPL","HD","PG",
                    "BA","INTC","TRV",
                    "CAT","IBM","UNH",
                    "CVX","JNJ","RTX",
                    "CSCO","JPM","VZ",
                    "KO","MCD","V",
                    "DD","MRK","WMT",
                    "XOM","MSFT","DIS"] 
                    # GM sold new shares around 2009 and the old share prices are not published online

#%%
for name in tickersDJIA2016:
    tickerInfo = yf.Ticker(name)
    f = open("/home/nikap/Desktop/Masterthesis/Conic_Portfolio/data/DJ2015/"+name+"_2015_stock.csv", "a")
    f.write(pd.DataFrame.to_csv(tickerInfo.history(start=startDate,end=endDate)))
    f.close()
#%% Set start and end times 2020
# startDate , as per our convenience we can modify
startDate = datetime.datetime(2017, 2, 15)
 
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2021, 2, 15)

# Get stock tickers
tickersDJIA2020 = ["MMM","GS","PFE",
                    "AXP","HD","PG",
                    "AAPL","INTC","TRV",
                    "BA","IBM","UNH",
                    "CAT","JNJ","RTX",
                    "CVX","JPM","VZ",
                    "CSCO","MCD","V",
                    "KO","MRK","WBA",
                    "DD","MSFT","WMT",
                    "XOM","NKE","DIS"] 
                    # GM sold new shares around 2009 and the old share prices are not published online

#%%
for name in tickersDJIA2020:
    tickerInfo = yf.Ticker(name)
    f = open("/home/nikap/Desktop/Masterthesis/Conic_Portfolio/data/DJ2020/"+name+"_2020_stock.csv", "a")
    f.write(pd.DataFrame.to_csv(tickerInfo.history(start=startDate,end=endDate)))
    f.close()
# %%
