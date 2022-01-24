# import Pkg; # We are going to install the packages CSV and DataFrames
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Plots")
# Pkg.add("JuMP")
# Pkg.add("Ipopt")
# Pkg.add("AmplNLWriter")
using CSV
using DataFrames
using Statistics
using Plots
using JuMP # language
using AmplNLWriter # interface
using Ipopt # solver
# cd("C:\\Users\\nikap\\Desktop\\Conic_Portfolio")
cd("/home/nikap/Desktop/Masterthesis/Conic_Portfolio") # On Linux


struct Asset
    name::String    
    returns::Vector{Float64}
    mean::Float64
    vol::Float64
end

# Now let's read in the stock information
csv_brk = CSV.File("BRK-B.csv");
csv_nke = CSV.File("NKE.csv");
csv_fb = CSV.File("FB.csv");
csv_v = CSV.File("V.csv");
csv_googl = CSV.File("GOOGL.csv");

df_brk = DataFrame(csv_brk) # NIKE(NKE) GOOGLE(GOOGL) VISA(V) META(FB) BERKSHIRE(BRK-B)
df_nke = DataFrame(csv_nke)
df_fb = DataFrame(csv_fb)
df_v = DataFrame(csv_v)
df_googl = DataFrame(csv_googl)

df_list = [df_brk,df_nke,df_fb,df_v,df_googl];

# google_closes = df_googl."Adj Close";
# google_returns = diff(google_closes)./google_closes[1:end-1]
# brk_closes = df_brk."Adj Close";
# brk_returns = diff(brk_closes)./brk_closes[1:end-1]
# fb_closes = df_fb."Adj Close";
# fb_returns = diff(fb_closes)./fb_closes[1:end-1]
# v_closes = df_v."Adj Close";
# v_returns = diff(v_closes)./v_closes[1:end-1]
# nke_closes = df_nke."Adj Close";
# nke_returns = diff(nke_closes)./nke_closes[1:end-1]
uniqueNames=["BRK-B","NKE","FB","V","GOOGL"];
nrAssets = length(df_list);
AssetArray = Array{Asset}(undef,nrAssets);
for i = 1:nrAssets
    df = df_list[i]
    name = uniqueNames[i]
    closes = df."Adj Close"
    returns = diff(closes)./closes[1:end-1]
    gemiddelde = 250*mean(returns)
    volatiliteit = sqrt(250)*std(returns)
    AssetArray[i] = Asset(name,returns,gemiddelde,volatiliteit)
end
AssetArray
ass = AssetArray[1]
println("Name: ",ass.name,"\nMean: ",ass.mean,"\nVol: ",ass.vol)

means(a::Vector{Asset}) = Vector{Float64}([ass.mean for ass in a])
vols(a::Vector{Asset}) = Vector{Float64}([ass.vol for ass in a])
returns(a::Vector{Asset}) = Vector{Vector{Float64}}([ass.returns for ass in a])

plot(vols(AssetArray), means(AssetArray),seriestype=:scatter)

# lets start with the optimization calculations.
# We need the following: 
# minimize          w⋅(Σw)
# subject to        w⋅μ = μ_required

# let us discretize a bunch of values for μ_required
μ_required = range(minimum(means(AssetArray)),stop=maximum(means(AssetArray)),length=100);
σ_efficient = zeros(100);

# and also acquire the Σ matrix and μ vector
Σ = 250*cov(hcat(returns(AssetArray)...));
μ = means(AssetArray);


# optimization
model = Model();
set_optimizer(model, Ipopt.Optimizer)
@variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
@constraint(model, lt1, sum(w) == 1);
@objective(model, Min, w'*Σ*w);
for i = 1:100
    println(i);
    μ_i = μ_required[i];
    @constraint(model, con, transpose(w)*μ == μ_i);
    optimize!(model);
    σ_efficient[i] = objective_value(model);
    delete(model,con);
    unregister(model,:con);
end

scatter(vols(AssetArray).^2, means(AssetArray),seriestype=:scatter,series_annotations = text.((uniqueNames), :left),xlabel="Volatility σ",ylabel="Return μ",label="Stocks",title="Efficient Frontier",legend=:bottomright)
plot!(σ_efficient,μ_required,linewidth=2,thickness_scaling=1,seriescolor=:blue,label="Efficient Frontier")

nrSamplePortfolios = 10
σ_samples = zeros(nrSamplePortfolios)
μ_samples = zeros(nrSamplePortfolios)
for i = 1:nrSamplePortfolios
    w = rand(nrAssets) #-repeat([0.5],nrAssets)
    w = w/sum(w)
    σ_samples[i] = w'*Σ*w
    μ_samples[i] = w'*μ
end
plot!(σ_samples,μ_samples,seriestype=:scatter,seriescolor=:yellow,label="Sample portfolios") # s&p 500


model = Model();
set_optimizer(model, Ipopt.Optimizer)
@variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
@constraint(model, lt1, sum(w) == 1);
@objective(model, Min, w'*Σ*w);
optimize!(model);
σ_minimum = objective_value(model);
w_optimMPT = value.(w)
μ_minimum = w_optimMPT'*means(AssetArray)

plot!([σ_minimum],[μ_minimum],seriestype=:scatter,linewidth=2,seriescolor=:black,label="Global Minimum Variance Portfolio")