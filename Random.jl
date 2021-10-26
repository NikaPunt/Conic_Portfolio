# import Pkg; # We are going to install the packages CSV and DataFrames
# Pkg.add("CSV")
# Pkg.add("DataFrames")
using CSV
using DataFrames
using Statistics
using Plots
cd("/home/nikap/Desktop/individual_stocks_5yr")


struct Asset
    name::String    
    returns::Vector{Float64}
    mean::Float64
    vol::Float64
end

# Now let's read in the stock information
csv_reader = CSV.File("all_stocks_5yr.csv")
df = DataFrame(csv_reader)
uniqueNames = unique(csv_reader.Name) 
nrAssets = length(470) #This would've been uniqueNames if every asset was recorded at the same time
AssetArray = Array{Asset}(undef,nrAssets)
for i = 1:nrAssets
    name = uniqueNames[i]
    A = df[df.Name .== name,:]
    if length(A.close) == 1259
        returns = diff(A.close)./A.close[1:end-1]
        gemiddelde = mean(returns)
        volatiliteit = std(returns)
        AssetArray[i] = Asset(name,returns,gemiddelde,volatiliteit)
    end
end

ass = AssetArray[433]
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
μ_required = range(-0.004,stop=0.003,length=1000)

# and also acquire the Σ matrix
Σ = var(hcat(returns(AssetArray)...))












B = unique([length(a) for a in returns(AssetArray)])

A = Vector{Int64}(zeros(29))
for i = 1:length(AssetArray)
    lengte = length(AssetArray[i].returns)
    index = findall(B.==Int64(lengte))[1] 
    A[index] = Int64(A[index]) + 1
end