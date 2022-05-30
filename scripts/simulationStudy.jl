include("HeaderFile.jl")
println("Importing datasets")
using Dates
using HDF5, JLD
using JLD2, FileIO
begin
    filenames = readdir("data/StudyStocks")
    N = length(filenames)-1
    uniqueNames = Vector{String}(undef,N)
    for i = 1:N
        name = filenames[i]
        coin = name[1:end-4]
        uniqueNames[i] = coin
    end



    df_list = Vector{DataFrame}(undef,N)
    for i = 1:N
        df_list[i] = DataFrame(CSV.File("data/StudyStocks/"*filenames[i],delim=","))
    end

    #Let's get all the dates for BTC (longest)
    dates = df_list[2]."Date"
    for i = 1:N
        dates = intersect(dates,df_list[i]."Date")
    end

    for i = 1:N # make sure we have all dates
        df_list[i] = filter(row -> row.:Date in dates, df_list[i])
    end
end

allDates = df_list[1]."Date"[251:3744]
for run = 1:100
    println("----------run $run------------$(Dates.now())--------")
    # for i = 1:10
        # println("This is simulation part $i out of 10") 
    df_list2 = unique(rand(df_list,rand(5:20)));

    nrAssets = length(df_list2)
    assetIndices = [findall(x->x==df_list2[i], df_list)[1] for i = 1:nrAssets]

    theDate = rand(allDates)
    y2k_pos = findall(x->x==theDate,df_list[1]."Date")[1]
    println("----------Date $theDate------------nr assets $nrAssets----")
    println("Calculating daily returns; ")

    nrCalibrationDays = 250 
    nrRealizedReturnDays = 30

    sampleRtrns = zeros(nrAssets,nrCalibrationDays) #matrix containing the daily returns of each asset in each row.
    sampleCrashRtrns = zeros(nrAssets,nrRealizedReturnDays)
    assetShiftedRtrns = Array{Float64,2}(undef,nrAssets,nrCalibrationDays); #same as up here but then making mean = 0
    AssetArray = Array{Asset}(undef,nrAssets);
    for i = 1:nrAssets
        df = df_list2[i]
        name = uniqueNames[i]
        closes = df."Close"
        rtrns = log.(closes[2:end]./closes[1:end-1])
        beforeCrashRtrns = rtrns[y2k_pos-nrCalibrationDays+1:y2k_pos]
        afterCrashRtrns = rtrns[y2k_pos+1:y2k_pos+nrRealizedReturnDays]
        sampleRtrns[i,:] = beforeCrashRtrns
        sampleCrashRtrns[i,:] = afterCrashRtrns
        gemiddelde = mean(beforeCrashRtrns)
        shiftedRtrns = beforeCrashRtrns-repeat([gemiddelde],length(beforeCrashRtrns))
        assetShiftedRtrns[i,:] = shiftedRtrns
        volatiliteit = sqrt(30.437)*std(beforeCrashRtrns)
        AssetArray[i] = Asset(name,beforeCrashRtrns,gemiddelde,volatiliteit)
    end


    print("MPT long; ")
    w_optimMPT = getMinVolWeights(AssetArray,false)
    print("MPT short; ")
    w_optimMPT_short = getMinVolWeights(AssetArray,true)
    print("Calculating returns; ")
    Rtrns = simulateJointReturns(assetShiftedRtrns, 10000)
    if ismissing(Rtrns)
        println("Going to next run: ")
        continue
    end
    print("CVaR 95% long; ")
    w_optimcvar95 = getMinCVaRWeights(Rtrns,0.95,false)
    print("CVaR 95% short; ")
    w_optimcvar95_short = getMinCVaRWeights(Rtrns,0.95,true)
    print("CVaR 99% long; ")
    w_optimcvar99 = getMinCVaRWeights(Rtrns,0.99,false)
    print("CVaR 99% short; ")
    w_optimcvar99_short = getMinCVaRWeights(Rtrns,0.99,true)
    print("VaR 95% long; ")
    w_optimvar = getMinVaRWeights(Rtrns,0.95,false)
    print("VaR 95% short; ")    
    w_optimvar_short = getMinVaRWeights(Rtrns,0.95,true)

    a_double = vec([0.14 0.4 1.0 2.6 5.0 0.14 0.4 1.0 2.6 5.0])
    # five_γ = Vector{Vector{Float64}}(undef,5)
    # five_γ_short = Vector{Vector{Float64}}(undef,5)
    all_γ = Vector{Vector{Float64}}(undef,10)
    Threads.@threads for i = 1:10
        id = Threads.threadid()
        println("Conic $(a_double[i]) on thread $id on iteration $i")
        nummer = a_double[i] 
        if i < 6
            all_γ[i] = getMinConicWeights(Rtrns,a_double[i],false)
        else
            all_γ[i] = getMinConicWeights(Rtrns,a_double[i],true)
        end
    end
    # five_γ = all_γ[1:5]
    # five_γ_short = all_γ[6:10]

    d = Dict(
    "theDate" => theDate,
    "y2k_pos" => y2k_pos,
    "assetIndices" => assetIndices,
    "w_optimMPT" => w_optimMPT,
    "w_optimMPT_short" => w_optimMPT_short,
    "w_optimvar" => w_optimvar,
    "w_optimvar_short" => w_optimvar_short,
    "w_optimcvar95" => w_optimcvar95,
    "w_optimcvar95_short" => w_optimcvar95_short,
    "w_optimcvar99" => w_optimcvar99,
    "w_optimcvar99_short" => w_optimcvar99_short,
    "all_γ" => all_γ
    )
    save("study/$(floor(Int,Dates.datetime2unix(Dates.now())))/data.jld2", "data", d)
end






# folders = readdir("study/")

# d3 = Dict{String,Any}

# for folder in folders
#     d3 = load("study/"*folder*"/data.jld2")["data"]
# end
# d3["all_γ"]

