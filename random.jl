cd("/home/nikap/Desktop/Masterthesis/Conic_Portfolio") # On Linux
using CSV
using DataFrames
using Statistics
using Plots
using IndependentComponentAnalysis
using Loess
include("ICA_assets.jl")
include("options.jl")
include("implied_moments.jl")

csv_brk = CSV.File("BRK-B.csv");
csv_brk_calls = CSV.File("brkb_calls.csv");
csv_brk_puts = CSV.File("brkb_puts.csv");
csv_nke = CSV.File("NKE.csv");
csv_nke_calls = CSV.File("nke_calls.csv");
csv_nke_puts = CSV.File("nke_puts.csv");
csv_fb = CSV.File("FB.csv");
csv_fb_calls = CSV.File("fb_calls.csv");
csv_fb_puts = CSV.File("fb_puts.csv");
csv_v = CSV.File("V.csv");
csv_v_calls = CSV.File("v_calls.csv");
csv_v_puts = CSV.File("v_puts.csv");
csv_googl = CSV.File("GOOGL.csv");
csv_googl_calls = CSV.File("googl_calls.csv");
csv_googl_puts = CSV.File("googl_puts.csv");

df_brk = DataFrame(csv_brk) # NIKE(NKE) GOOGLE(GOOGL) VISA(V) META(FB) BERKSHIRE(BRK-A)
df_brk_calls = DataFrame(csv_brk_calls)
df_brk_puts = DataFrame(csv_brk_puts)
df_nke = DataFrame(csv_nke)
df_nke_calls = DataFrame(csv_nke_calls)
df_nke_puts = DataFrame(csv_nke_puts)
df_fb = DataFrame(csv_fb)
df_fb_calls = DataFrame(csv_fb_calls)
df_fb_puts = DataFrame(csv_fb_puts)
df_v = DataFrame(csv_v)
df_v_calls = DataFrame(csv_v_calls)
df_v_puts = DataFrame(csv_v_puts)
df_googl = DataFrame(csv_googl)
df_googl_calls = DataFrame(csv_googl_calls)
df_googl_puts = DataFrame(csv_googl_puts)

df_list = [df_brk,df_nke,df_fb,df_v,df_googl];
yields = [0,0.0078,0,0.0069,0];
calls_list = df2option.([df_brk_calls,df_nke_calls,df_fb_calls,df_v_calls,df_googl_calls],call)
puts_list = df2option.([df_brk_puts,df_nke_puts,df_fb_puts,df_v_puts,df_googl_puts],put)



print(calls_list[5].strikes[calls_list[5].strikes .< 2000])
plot(calls_list[1].strikes,calls_list[1].prices)
plot(puts_list[5].strikes,puts_list[5].prices)
print(puts_list[1].prices)

nrAssets = length(df_list);
assetMeans = Vector{Float64}(undef,nrAssets);
assetShiftedReturns = Array{Float64,2}(undef,nrAssets,length(df_brk."Adj Close")-1);
for i = 1:nrAssets
    df = df_list[i]
    closes = df."Adj Close"
    returns = log.(closes[2:end]./closes[1:end-1])
    gemiddelde = mean(returns)
    shiftedReturns = returns-repeat([gemiddelde],length(returns))
    assetShiftedReturns[i,:] = shiftedReturns;
end

ica_comps = ICA_assets(assetShiftedReturns)

# X = A*S      where X are the returns, A is mixing matrix and S is independent comps
# S = W'*X
W = ica_comps.W
A = ica_comps.mixing
S = ica_comps.indcomps

include("implied_moments.jl")
include("getVGParams.jl")
for i = 1:5
    pars = get_params(df_list[i]."Adj Close"[end],1,yields[i],0.004,calls_list[i],puts_list[i])
    print(getVGParams(pars,1),"\n")
end