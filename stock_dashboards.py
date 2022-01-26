import streamlit as st
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os

#sidebar
st.sidebar.title("Dashboards")
st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 14rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 14rem;}}
    </style>
''',unsafe_allow_html=True)

dashboards = ("Overview Charts", "Stock Analysis", "Stock Prediction", "Porfolio Returns")
option = st.sidebar.selectbox("Select A Dashboard", dashboards, 1)

#page header
st.header(option.upper())
today = datetime.date.today()
stock_symbol = ("AAPL", "GOOG", "AMZN", "MSFT", "TSLA", "IVV", "MSCI", "VOO")

#function to load stock data from yfinance
@st.experimental_memo(ttl=24*3600)
def load_data(ticker, start, today):
    data = yf.download(ticker, start, today)
    return data 

#DASHBOARD 1: OVERVIEW CHARTS
if option == "Overview Charts":
    st.markdown("Import charts from [finviz.com](https://finviz.com/) to have an overview of interested stocks in the last 6 months")
    for stock in stock_symbol:
        st.image(f"https://finviz.com/chart.ashx?t={stock}")

#DASHBOARD 2: STOCK ANALYSIS
if option == "Stock Analysis":
    st.markdown("Import raw data from [Yahoo Finance](https://finance.yahoo.com/) and run analyses on the selected stocks. ")
    years_analysis = st.slider("Adjust years of past data:", 1, 10, 1)
    start_analysis = today - datetime.timedelta(years_analysis * 365)

    #load data from yfinance
    data_analysis = load_data(stock_symbol, start_analysis, today)
    data_analysis = data_analysis.swaplevel(0,1,axis=1)
    data_analysis = data_analysis.dropna()
    close_price = data_analysis.xs(key='Adj Close',axis=1,level=1)

    #plot closing price
    def plotly_chart(df):
        fig = px.line(df,
                      x=df.index,
                      y=df.columns,
                      title='Closing Price Last 12 Months')
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
    
    plotly_chart(close_price)

    ### RETURNS ANALYSIS
    st.subheader("Daily Return Analysis")
    returns = close_price.pct_change()
    st.write(f'Caculate Daily Returns Last {years_analysis} Year(s). Show Last 7 Working Days')
    st.write(returns.tail(7).sort_index(ascending=False))
    
    #correlation heatmap on returns
    def heat_map(df):
        corr = df.corr()
        mask = np.triu(np.ones_like(corr,dtype=bool))
        dta = go.Heatmap(z=corr.mask(mask),
                        x=corr.columns,
                        y=corr.columns,
                        colorscale=px.colors.diverging.RdBu,                        
                        zmin=-1,
                        zmax=1)
        fig = go.Figure(data=[dta])
        fig.update_xaxes(side="bottom")
        fig.update_layout(title_text=f'Correlation of Daily Returns Last {years_analysis} Years',
                            width=700, height = 500,
                            xaxis_showgrid=False,
                            yaxis_showgrid=False,
                            yaxis_autorange='reversed',
                            template='plotly_white',
                            autosize=True)        
        st.plotly_chart(fig, use_container_width=True)

    heat_map(returns)

    #stock clustering chart by returns
    def cluster_map(df):
        corr = df.corr()
        fig = ff.create_dendrogram(corr.to_numpy(),
                                    labels=corr.columns.tolist(),
                                    orientation='left')
        fig.update_layout(title_text=f"Stock Clustering Based on Returns Last {years_analysis} Years")                                  
        st.plotly_chart(fig, use_container_width=True)
    
    cluster_map(returns)
    
    #correlation between 2 stocks - scatter plot
    st.write("More on Correlation Between 2 Selected Stocks")

    box_1 = st.selectbox("Select Stock 1", stock_symbol, 1)
    box_2 = st.selectbox("Select Stock 2", stock_symbol, 3)

    def scatter_plot(df, stock1, stock2):
        fig = px.scatter(df,x=f"{stock1}",
                            y=f"{stock2}",
                            marginal_x="histogram",
                            marginal_y="rug",
                            trendline = 'ols')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    scatter_plot(returns, box_1, box_2)

    ### INDICATORS
    st.subheader("Indicators")
    chosen_single = st.selectbox("Choose a symbol", stock_symbol)
    single = data_analysis.xs(key=f'{chosen_single}', axis=1)    

    #choose period for the indicators
    coIN1, coIN2 = st.columns(2)
    with coIN1:
        numYear = st.number_input('Insert period (Year) 1-10: ', min_value=1, max_value=10, value=1, key=0)    
    with coIN2:
        windowSize = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=21, key=1)
    
    #load data
    start_indicator = datetime.datetime.today()-datetime.timedelta(numYear*365)
    end_indicator = datetime.datetime.today()
    data_indicator = load_data(chosen_single,start_indicator,end_indicator)   

    #define SMA and Bollinger indicators
    def calcMovingAverage(data, size):
        df = data.copy()
        df['sma'] = df['Adj Close'].rolling(size).mean()
        df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
        df.dropna(inplace=True)
        return df
    
    def calcBollinger(data, size):
        df = data.copy()
        df["sma"] = df['Adj Close'].rolling(size).mean()
        df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
        df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df

    def plot_sma(df):         
        df_ma = calcMovingAverage(df, windowSize)
        df_ma = df_ma.reset_index()

        figMA = go.Figure()    
        figMA.add_trace(go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['Adj Close'],
                        name = "Close Price"))    
        figMA.add_trace(go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(windowSize)))
        figMA.update_layout(legend=dict(yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01))
        figMA.add_trace(go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(windowSize)))    
        figMA.update_layout(legend_title_text='Trend',
                            title_text = "SMA: Simple Moving Average Indicator")
        figMA.update_yaxes(tickprefix="$")    
        st.plotly_chart(figMA, use_container_width=True)
    
    def plot_bollinger(df):
        df_boll = calcBollinger(df, windowSize)
        df_boll = df_boll.reset_index()
        figBoll = go.Figure()
        figBoll.add_trace(go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bolu'],
                            name = "Upper Band"))      
        figBoll.add_trace(go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['sma'],
                            name = "SMA" + str(windowSize)))        
        figBoll.add_trace(go.Scatter(
                            x = df_boll['Date'],
                            y = df_boll['bold'],
                            name = "Lower Band"))       
        figBoll.update_layout(legend=dict(orientation="h",
                                            yanchor="bottom",
                                            y=1,
                                            xanchor="left",
                                            x=0), title_text = "Bollinger Bands Indicator")        
        figBoll.update_yaxes(tickprefix="$")
        st.plotly_chart(figBoll, use_container_width=True)

    #plot indicators
    plot_sma(data_indicator)
    plot_bollinger(data_indicator)

    ### RISK ANALYSIS    
    #plot expected returns vs std of daily returns = risk
    st.subheader("Risk Analysis")
    st.write(f"Using data last {years_analysis} year(s) from Yahoo Finance. To adjust the timeframe, use the slider on top of the page.")

    def plotly_risk(df):
        fig = px.scatter(x=df.mean(), y=df.std(),
                        labels=dict(x='Expected Return', y='Risk'),
                        text=df.columns)
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)
        fig.update_traces(marker_size=20,
                            marker_color=px.colors.sequential.Emrld,
                            textposition='top center')
        fig.update_layout(title_text='Quantify Risk by Comparing Expected Return with Standard Deviation of Daily Returns')
        st.plotly_chart(fig, use_container_width=True)

    plotly_risk(returns)

    #Monte Carlo Simulation & Value at Risk (VaR)
    chosen_var = st.selectbox("Select a symbol for Monte Carlo Simulations & VaR Analysis", stock_symbol)
    data_var = data_analysis.xs(key=f'{chosen_var}', axis=1)
    returns_var = returns[f'{chosen_var}']

    #Monte Carlo simulation
    days=365
    dt=1/days
    mu = returns_var.mean()
    sigma = returns_var.std()
    start_price = data_var['Open'].head(1)

    def stock_monte_carlo(start_price,days,mu,sigma):    
        price = np.zeros(days)
        price[0] = start_price
        shock = np.zeros(days)
        drift = np.zeros(days)
        
        for x in range(1,days):            
            shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            drift[x] = mu * dt
            price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))            
        return price  
        
    #plot simulations
    fig, ax = plt.subplots()
    for run in range(50):
        ax.plot(stock_monte_carlo(start_price,days,mu,sigma))
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.set_autoscale_on(True)
    ax.grid(False)
    st.write("Monte Carlo Simulations")
    st.pyplot(fig)

    #quantile plot to define VaR    
    runs=700
    simulations = np.zeros(runs)
    np.set_printoptions(threshold=5)

    for run in range(runs):    
        simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
    
    q = np.percentile(simulations, 1)
    var_plot, ax = plt.subplots()
    ax.hist(simulations,bins=150, color='blue')
    ax.axvline(x=q, linewidth=3, color='r')
    ax.set_autoscale_on(True)
    ax.grid(False)    

    st.write(f"Value at Risk & Final Price Distribution for {chosen_var} Last 365 Days")
    st.pyplot(var_plot)

    #display metrics for VaR plot
    col3, col4 = st.columns(2)
    col3.metric(label="q(0.99) (the red line)", value="$%.2f" %q)
    col4.metric(label="VaR(0.99)", value="$%.2f" %(start_price-q))
    col1, col2 = st.columns(2)
    col1.metric(label="Start price", value="$%.2f" %start_price)
    col2.metric(label="Mean final price", value="$%.2f" %simulations.mean())

    st.caption("Select other dashboards on the sidebar for more!")
    st.markdown("_Language: Python. [Go to my source code](https://github.com/uyenphan48/streamlit_dashboards/blob/main/stock_dashboards.py)_")     

#DASHBOARD 3: STOCK PREDICTION
if option == "Stock Prediction":
    st.markdown("Import raw data from [Yahoo Finance](https://finance.yahoo.com/) and run prediction on the selected stock. ")
    stocks = ("AAPL", "GOOG", "AMZN", "MSFT", "TSLA", "IVV", "VOO", "MSCI", "GC=F", "BTC-USD") 
    selected_stocks = st.selectbox("Select stock for prediction", stocks)
    symbol_predict = st.text_input("Other: enter stock symbol", value="")

    if symbol_predict:
        ticker = f"{symbol_predict}"
    else:
        ticker = selected_stocks   

    #year range for historical data
    his_years = st.slider("Years of historical data:", 1, 10, 3)
    days = datetime.timedelta(his_years*365)
    start_predict = today - days

    #year range for prediction
    n_years = st.slider("Years of prediction:", 1, 10)
    input_period = n_years * 365
         
    #load data from yfinance
    data_load_state = st.text("Load data...")
    data_yahoo = load_data(ticker, start_predict, today)
    data_yahoo.reset_index(inplace=True)
    data_load_state.text("Loading data... done!")

    #print dataframe last 7D
    st.subheader('Histocial Data')
    st.write("Display Raw Data Last 7D")
    st.write(data_yahoo.tail(7).sort_values (by='Date', ascending=False))

    #candlestick chart last 3 months
    delta_month = today - relativedelta(months=3)
    data_month = load_data(ticker, delta_month, today)
    data_month.reset_index(inplace=True)

    def candlestick(data, title):
        fig = go.Figure(data = [go.Candlestick(x=data['Date'],
                                            open = data['Open'],
                                            high = data['High'],
                                            low = data['Low'],
                                            close = data['Close'],
                                            name=ticker)])                                            
        fig.update_layout(title_text=title, height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    candle_title = "Candlestick Chart for Last 3 Months"
    candlestick(data_month, candle_title)

    #plot time series of selected historical data
    def plot_raw_data(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'],
                                y=df['Open'],
                                name='stock_open'))
        fig.add_trace(go.Scatter(x=df['Date'],
                                y=df['Close'],
                                name='stock_close'))
        fig.layout.update(title_text="All Data for Selected Time", height=600, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw_data(data_yahoo)
    
    ### FORCASTING
    @st.experimental_memo(ttl=24*3600)
    def m_forecast(df):
        df_train = df[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet()
        m.fit(df_train)
        return m
    
    @st.experimental_memo(ttl=24*3600)
    def forecasting(df, input_period):
        m = m_forecast(df)
        future = m.make_future_dataframe(periods=input_period)
        forecast = m.predict(future)
        return forecast

    m = m_forecast(data_yahoo)
    forecast_state = st.text("Load forecast data...")
    forecast = forecasting(data_yahoo, input_period)
    forecast_state.text("")

    #display forecast data 7 days
    st.subheader('Forcasting')
    st.write(forecast.tail(7).sort_values(by='ds', ascending=False))

    #plot the forecast
    fig1 = plot_plotly(m, forecast)
    fig1.layout.update(title_text="Interactive Forcast Chart", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1, use_container_width=True)

    #plot other forecase components
    st.write('Forecast Component')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

#DASHBOARD 4: PORFOLIO RETURNS
if option == "Porfolio Returns":
    #connect to alpaca API
    load_dotenv()
    key_id = os.getenv("APCA_API_KEY_ID")
    secret_key= os.getenv("APCA_API_SECRET_KEY")
    api = tradeapi.REST(key_id, secret_key, base_url='https://paper-api.alpaca.markets')    

    def alpaca_table(list):
        df_2 = pd.DataFrame()
        for item in list:
            to_dict = vars(item)
            df = pd.DataFrame.from_dict(to_dict, orient='index')
            df_2 = df_2.append(df, ignore_index=True)
        return df_2

    #get trading history from alpaca
    activities = api.get_activities(activity_types="FILL")
    df_2 = alpaca_table(activities)
    df_2 = df_2[['transaction_time', 'symbol', 'side', 'qty', 'price', 'order_status', 'cum_qty']]
    df_2['transaction_time'] = pd.to_datetime(df_2['transaction_time'], utc=True)
    df_2['price'] = df_2['price'].astype('float')
    df_2[['qty', 'cum_qty']] = df_2[['qty', 'cum_qty']].astype('int32')
   
    st.subheader("Trading Log")
    st.markdown("Use [Alpaca](https://alpaca.markets/) API to retrieve transaction history from my [Alpaca Paper Trading](https://alpaca.markets/docs/trading-on-alpaca/paper-trading/) account")
    st.write(df_2)

    #get porfolio values
    port = api.list_positions()
    df_3 = alpaca_table(port)    
    df_3 = df_3[['symbol', 'qty', 'market_value', 'unrealized_pl', 'current_price', 'avg_entry_price', 'cost_basis', 'unrealized_plpc', 'side']]
    df_3['qty'] = df_3['qty'].astype('int')
    df_3['unrealized_plpc'] = df_3['unrealized_plpc'].astype(float).apply(lambda x: x*100)
    columns = ['avg_entry_price', 'current_price', 'cost_basis', 'market_value', 'unrealized_pl']
    df_3[columns] = df_3[columns].astype('float')    

    st.subheader("Porfolio Overview")
    st.write("Get an overview of Porfolio Values to date through Alpaca API (using paper trading)")
    
    #display PL metrics
    port_value = df_3['market_value'].sum()
    sum_cost = df_3['cost_basis'].sum()
    sum_earn = df_3['unrealized_pl'].sum()
    max_lose = df_3['unrealized_pl'].min()
    max_lose_name = df_3[df_3['unrealized_pl']==max_lose]['symbol'].values[0]
    max_win = df_3['unrealized_pl'].max()
    max_win_name = df_3[df_3['unrealized_pl']==max_win]['symbol'].values[0]
    delta_portvalue = ((port_value - sum_cost) / sum_cost)*100
    delta_maxlose = (max_lose / df_3[df_3['unrealized_pl']==max_lose]['cost_basis'].values[0])*100
    delta_maxwin = (max_win / df_3[df_3['unrealized_pl']==max_win]['cost_basis'].values[0])*100

    col1, col2 = st.columns(2)
    col1.metric(label="Total Values", value="$%.2f" %port_value, delta=" %.3f" % delta_portvalue + "%")
    col2.metric(label="Total P/L to Date", value="$%.2f" %sum_earn, delta=" %.3f" % delta_portvalue + "%")

    col3, col4 = st.columns(2)
    col3.metric(label="Max Losing Stock", value=max_lose_name + " $%.2f" %max_lose, delta=" %.3f" % delta_maxlose + "%")
    col4.metric(label="Max Earning Stock", value=max_win_name + " $%.2f" %max_win, delta=" %.3f" % delta_maxwin + "%")

    st.dataframe(df_3)

    #plot propotion of stocks
    def donut_chart(df):
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        df['qty_percent'] = round((df['qty']/df['qty'].sum())*100,2)
        df['cost_percent'] = round((df['cost_basis']/df['cost_basis'].sum())*100,0)
        
        fig.add_trace(go.Pie(labels=df['symbol'], values=df['qty_percent'],), 1, 1)
        fig.add_trace(go.Pie(labels=df['symbol'], values=df['cost_percent']), 1, 2)

        fig.update_traces(hole=.4, hoverinfo="label+percent")
        fig.update_layout(title_text="Proportion of Stock Quantity & Cost",
                          annotations=[dict(text='Quantity', x=0.15, y=0.5, font_size=20, showarrow=False),
                                       dict(text='Cost', x=0.81, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig)
    
    st.subheader("Portfolio Breakdown")
    donut_chart(df_3)








        



