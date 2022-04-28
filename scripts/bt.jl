#Bring everything together

include("HeaderFile.jl")
println("Importing datasets")
begin
    crypto_filenames = readdir("datasets/crypto10")
    const N = length(crypto_filenames)
    uniqueNames = Vector{String}(undef,N)
    for i = 1:N
        name = crypto_filenames[i]
        coin = name[9:end-10]
        uniqueNames[i] = coin
    end

    df_list = Vector{DataFrame}(undef,N)
    for i = 1:N
        df_list[i] = DataFrame(CSV.File("datasets/crypto10/"*crypto_filenames[i],delim=",",header=2))
    end

    #Let's get all the dates for BTC (longest)
    dates = df_list[2]."date"
    for i = 1:N
        dates = intersect(dates,df_list[i]."date")
    end

    for i = 1:N
        df_list[i] = filter(row -> row.:date in dates, df_list[i])
    end
end

println("Calculating daily returns")
begin
    nrAssets = length(df_list); #number of assets
    sampleRtrns = zeros(nrAssets,length(df_list[1]."close")-1) #matrix containing the daily returns of each asset in each row.
    assetShiftedRtrns = Array{Float64,2}(undef,nrAssets,length(df_list[1]."close")-1); #same as up here but then making mean = 0
    AssetArray = Array{Asset}(undef,nrAssets);
    for i = 1:nrAssets
        df = df_list[i]
        name = uniqueNames[i]
        closes = df."close"
        rtrns = log.(closes[2:end]./closes[1:end-1])
        sampleRtrns[i,:] = rtrns
        gemiddelde = mean(rtrns)
        shiftedRtrns = rtrns-repeat([gemiddelde],length(rtrns))
        assetShiftedRtrns[i,:] = shiftedRtrns
        volatiliteit = sqrt(30.437)*std(rtrns)
        AssetArray[i] = Asset(name,rtrns,gemiddelde,volatiliteit)
    end
end

w_optimMPT = getMinVolWeights(AssetArray,false)
w_optimMPT_short = getMinVolWeights(AssetArray,true)

Rtrns = simulateJointReturns(assetShiftedRtrns)

w_optimvar = getMinVaRWeights(Rtrns,0.95,false)
w_optimvar_short = getMinVaRWeights(Rtrns,0.95,true)

w_optimcvar = getMinCVaRWeights(Rtrns,0.95,false)
w_optimcvar_short = getMinCVaRWeights(Rtrns,0.95,true)

a = range(0.1,stop=12.8,length=5)
five_γ = Vector{Vector{Float64}}(undef,5)
for i = 1:5
    five_γ[i] = getMinConicWeights(Rtrns,a[i],false)
end
five_γ_short = Vector{Vector{Float64}}(undef,5)
for i = 1:5
    five_γ_short[i] = getMinConicWeights(Rtrns,a[i],true)
end


