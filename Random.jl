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
csv_reader = CSV.File("all_stocks_5yr.csv");
df = DataFrame(csv_reader) # NIKE(NKE) GOOGLE(GOOGL) VISA(V) META(FB) BERKSHIRE(BRK.B)
# uniqueNames = unique(csv_reader.Name);
# f = open("Stock_names.txt", "w")
# print(f, uniqueNames)
# close(f)
# nrAssets = length(uniqueNames);
# AssetArray = Array{Asset}(undef,470); #This would've been uniqueNames if every asset was recorded at the same time
# j = 0;
# volume = zeros(470)
uniqueNames = ["NKE","GOOGL","V","FB","BRK.B"]
nrAssets = 5;
AssetArray = Array{Asset}(undef,5); #This would've been uniqueNames if every asset was recorded at the same time
j = 0;
volume = zeros(5)
for i = 1:nrAssets
    name = uniqueNames[i]
    A = df[df.Name .== name,:]
    if length(A.close) == 1259
        j += 1
        returns = diff(A.close)./A.close[1:end-1]
        gemiddelde = 250*mean(returns)
        volatiliteit = sqrt(250)*std(returns)
        AssetArray[j] = Asset(name,returns,gemiddelde,volatiliteit)
        volume[j] = A.volume[i]
    end
end
println(j)
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
μ_required = range(-0.001,stop=0.0025,length=100);
σ_efficient = zeros(100);

# and also acquire the Σ matrix and μ vector
Σ = cov(hcat(returns(AssetArray)...));
μ = means(AssetArray);


# optimization
model = Model();
set_optimizer(model, Ipopt.Optimizer)
@variable(model, w[1:470] >= 0) # you can unregister w through unregister(model, w)
@constraint(model, lt1, sum(w) <= 1);
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

plot(vols(AssetArray).^2, means(AssetArray),seriestype=:scatter)
plot!(σ_efficient,μ_required)

for i = 1:3
    w = rand(470)
    w = w/sum(w)
    plot!([w'*Σ*w],[w'*μ],seriestype=:scatter,seriescolor=:red)
end
w = volume/sum(volume);
plot!([w'*Σ*w],[w'*μ],seriestype=:scatter,seriescolor=:purple) # s&p 500