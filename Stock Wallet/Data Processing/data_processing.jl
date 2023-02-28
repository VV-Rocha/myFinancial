
module DataDownload

using MarketData
MD = MarketData

function download(symbol; p1=Dates.now()-Year(2), p2=Dates.now(), interval="1mo")
    ## Set download parameters
    parameters = MD.YahooOpt(period1=p1, period2=p2, interval=interval)
    ## Stock symbol
    timeseries = MD.yahoo(symbol, parameters)
    return timeseries
end

function zero_offset(series)
    return map(x - x[1], )
end

function data_sort(timeseries)
    time = datetime2unix.(DateTime.(timestamp(timeseries)))
    time = map.(-, time, time[1])
    
    return values(timeseries["Open"]),
          values(timeseries["High"]), 
          values(timeseries["Low"]),
          values(timeseries["Close"]),
          time,
          values(timeseries["Volume"])
end

function set_download(symbols; p1=Dates.now()-Year(2), p2=Dates.now(), interval="1mo")
    stocks = Dict{String, Matrix}()
    
    for symbol in symbols
        Open, High, Low, Close, Times, Volume = data_sort(download(symbol, p1=p1, p2=p2, interval=interval))
        #print(size(Open), " ", size(Close)," ",size(Times),"\n")
        stocks[symbol] = hcat(Open, High, Low, Close, Volume, Times) ## [Time, data]
    end
    return stocks
end

function dictionary2array(stocks, symbols)
    arr = zeros(size(stocks[symbols[1]])[1], size(stocks[symbols[1]])[2], size(symbols)[1])
    for symbol in range(1, size(symbols)[1], step=1)
         arr[:, :, symbol] = stocks[symbols[symbol]]
    end
    print(size(arr))
    return arr
end

export download
export set_download
end