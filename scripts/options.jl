# import Pkg
# Pkg.add("Loess")
using Loess

@enum OPTION call=1 put=2

struct options
    type::OPTION
    strikes::Vector{Float64}
    prices::Vector{Float64}
end

function df2option(df,type::OPTION)
    strikes = df."strike"
    prices = (df."bid" + df."ask")/2
    smoothPrices = smooth_option(strikes,prices)
    return options(type,strikes,smoothPrices)
end

function smooth_option(strikes,prices)
    xs = (strikes)
    ys = (prices)

    model = Loess.loess(xs, ys, span = 0.5)

    vs = predict(model, xs)
    return vs
end