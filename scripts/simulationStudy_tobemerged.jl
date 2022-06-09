include("HeaderFile.jl")
println("Importing datasets")
using Dates
using HDF5
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

allDates = df_list[1]."Date"[1001:3710];
for run = 1:500
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

    nrCalibrationDays = 1000 
    nrRealizedReturnDays = 30

    sampleRtrns = zeros(nrAssets,nrCalibrationDays) #matrix containing the daily returns of each asset in each row.
    sampleCrashRtrns = zeros(nrAssets,nrRealizedReturnDays)
    assetShiftedRtrns = Array{Float64,2}(undef,nrAssets,nrCalibrationDays); #same as up here but then making mean = 0
    AssetArray = Array{Asset}(undef,nrAssets);
    for i = 1:nrAssets
        df = df_list2[i]
        name = uniqueNames[i]
        closes = df."Close"
        rtrns = (closes[2:end] .- closes[1:end-1] ) ./ closes[1:end-1]
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
    Rtrns = nothing
    try
        Rtrns = simulateJointReturns(assetShiftedRtrns, 10000)
    catch
        println("Calculating returns failed; skipping run")
        continue
    end
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
    t_w_optimvar = @async getMinVaRWeights(Rtrns,0.95,false)
    print("VaR 95% short; ")    
    t_w_optimvar_short = @async getMinVaRWeights(Rtrns,0.95,true)

    a_double = vec([0.14 0.4 1.0 2.6 5.0 0.14 0.4 1.0 2.6 5.0])
    # five_γ = Vector{Vector{Float64}}(undef,5)
    # five_γ_short = Vector{Vector{Float64}}(undef,5)
    all_γ = Vector{Vector{Float64}}(undef,10)
    Threads.@threads for i = 1:10
        id = Threads.threadid()
        println("Conic $(a_double[i]) on thread $id on iteration $i")
        nummer = a_double[i] 
        if i < 6
            all_γ[i] = getMinConicWeights(sampleRtrns,Rtrns,a_double[i],false)
        else
            all_γ[i] = getMinConicWeights(sampleRtrns,Rtrns,a_double[i],true)
        end
    end
    # five_γ = all_γ[1:5]
    # five_γ_short = all_γ[6:10]
    w_optimvar = nothing
    w_optimvar_short = nothing
    if (istaskdone(t_w_optimvar) & istaskdone(t_w_optimvar_short))
        w_optimvar = fetch(t_w_optimvar)
        w_optimvar_short = fetch(t_w_optimvar_short)
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
        save("study1000/$(floor(Int,Dates.datetime2unix(Dates.now())))/data.jld2", "data", d)
    end
end

folders = readdir("study1000/")
sampleSize = length(folders)
dataList = Vector{Dict{String,Any}}(undef,length(folders))

for (i,folder) in enumerate(folders)
    dataList[i] = load("study1000/"*folder*"/data.jld2")["data"]
end


function data2herfindahl(data)
    result = Vector{Float64}(undef,9)
    prodexp(x) = (x=max.(x,0);prod(x.^x))
    result[1] = prodexp(data["w_optimMPT"])
    result[2] = prodexp(data["w_optimcvar95"])
    result[3] = prodexp(data["w_optimcvar99"])
    result[4] = prodexp(data["w_optimvar"])
    result[5] = prodexp(data["all_γ"][1])
    result[6] = prodexp(data["all_γ"][2])
    result[7] = prodexp(data["all_γ"][3])
    result[8] = prodexp(data["all_γ"][4])
    result[9] = prodexp(data["all_γ"][5])
    return result
end

function data2grossreturns(data,nrDays=60)
    weights = [data["w_optimMPT"],
                data["w_optimMPT_short"],
                data["w_optimcvar95"],
                data["w_optimcvar95_short"],
                data["w_optimcvar99"],
                data["w_optimcvar99_short"],
                data["w_optimvar"],
                data["w_optimvar_short"],
                data["all_γ"][1],
                data["all_γ"][6],
                data["all_γ"][2],
                data["all_γ"][7],
                data["all_γ"][3],
                data["all_γ"][8],
                data["all_γ"][4],
                data["all_γ"][9],
                data["all_γ"][5],
                data["all_γ"][10]
            ]
    indices = data["assetIndices"]
    y2k_pos = data["y2k_pos"]
    stockdfs = df_list[indices]
    allGrossRtrns = zeros(length(indices),nrDays)
    for i = 1:length(indices)
        df = stockdfs[i]
        closes = df."Close"
        rtrns = log.(closes[2:end]./closes[1:end-1])
        grossRtrns = rtrns[y2k_pos+1:y2k_pos+nrDays]
        allGrossRtrns[i,:] = grossRtrns
    end
    something = zeros(length(weights),nrDays)
    for (i,weight) in enumerate(weights)
        something[i,:] = weight'*allGrossRtrns
    end
    return something
end

function data2quarterlyreturn(data,nrDays=63)
    weights = [data["w_optimMPT"],
                data["w_optimMPT_short"],
                data["w_optimcvar95"],
                data["w_optimcvar95_short"],
                data["w_optimcvar99"],
                data["w_optimcvar99_short"],
                data["w_optimvar"],
                data["w_optimvar_short"],
                data["all_γ"][1],
                data["all_γ"][6],
                data["all_γ"][2],
                data["all_γ"][7],
                data["all_γ"][3],
                data["all_γ"][8],
                data["all_γ"][4],
                data["all_γ"][9],
                data["all_γ"][5],
                data["all_γ"][10]
            ]
    indices = data["assetIndices"]
    y2k_pos = data["y2k_pos"]
    stockdfs = df_list[indices]
    allGrossRtrns = zeros(length(indices))
    for i = 1:length(indices)
        df = stockdfs[i]
        closes = df."Close"
        allGrossRtrns[i] = (closes[y2k_pos+nrDays] .- closes[y2k_pos] ) ./ closes[y2k_pos]
    end
    something = zeros(length(weights))
    for (i,weight) in enumerate(weights)
        something[i] = weight'*allGrossRtrns
    end
    return something
end



function grossreturn2finalreturn(grossrtrn)
    expgrossrtrn = exp.(grossrtrn)
    for i = 2:size(expgrossrtrn,2)
        expgrossrtrn[:,i] = expgrossrtrn[:,i-1] .* expgrossrtrn[:,i]
    end
    return vec(expgrossrtrn[:,end])
end

function index2maxdrawdown(index)
    n = length(index)
    drawdowns = []
    i = 1
    currentpeak = index[i]
    currenttrough = index[i]
    while i <= n
        if index[i] > currentpeak
            append!(drawdowns,[[currentpeak,currenttrough]])
            currentpeak = index[i]
            currenttrough = index[i]
        end
        if index[i] < currenttrough
            currenttrough = index[i]
        end
        i = i + 1
    end
    append!(drawdowns,[[currentpeak,currenttrough]])

    #let us find the maximum of all the drawdowns
    maxdrawdown = 0
    for (i,currentdrawdown) = enumerate(drawdowns)
        percentualchange = (currentdrawdown[1]-currentdrawdown[2])/currentdrawdown[1]
        if percentualchange > maxdrawdown
            maxdrawdown = percentualchange 
        end   
    end
    return maxdrawdown * 100
end

function data2maxdrawdown(data,nrDays=63)
    weights = [data["w_optimMPT"],
                data["w_optimMPT_short"],
                data["w_optimcvar95"],
                data["w_optimcvar95_short"],
                data["w_optimcvar99"],
                data["w_optimcvar99_short"],
                data["w_optimvar"],
                data["w_optimvar_short"],
                data["all_γ"][1],
                data["all_γ"][6],
                data["all_γ"][2],
                data["all_γ"][7],
                data["all_γ"][3],
                data["all_γ"][8],
                data["all_γ"][4],
                data["all_γ"][9],
                data["all_γ"][5],
                data["all_γ"][10]
            ]
    indices = data["assetIndices"]
    y2k_pos = data["y2k_pos"]
    stockdfs = df_list[indices]
    allStockCloses = zeros(length(indices),nrDays)
    for i = 1:length(indices)
        df = stockdfs[i]
        closes = df."Close"
        allStockCloses[i,:] = closes[y2k_pos+1:y2k_pos+nrDays] / closes[y2k_pos+1]
    end
    something = zeros(length(weights),nrDays)
    for (i,weight) in enumerate(weights)
        something[i,:] = weight'*allStockCloses
    end
    return index2maxdrawdown.(eachrow(something))
end


folders = readdir("study1000/")
dataList = Vector{Dict{String,Any}}(undef,length(folders))
# d3 = Dict{String,Any}

for (i,folder) in enumerate(folders)
    dataList[i] = load("study1000/"*folder*"/data.jld2")["data"]
end

maxdrawdownQuantiles[[1,3,5,7,9,11,13,15,17],:]
maxdrawdownQuantiles[[2,4,6,8,10,12,14,16,18],:]

mddQDF = DataFrame(maxdrawdownQuantiles,:auto)
insertcols!(mddQDF,1,:method=>["w_optimMPT",
"w_optimMPT_short",
"w_optimcvar95",
"w_optimcvar95_short",
"w_optimcvar99",
"w_optimcvar99_short",
"w_optimvar",
"w_optimvar_short",
"conic 0.14",
"conic short 0.14",
"conic 0.4",
"conic short 0.4",
"conic 1.0",
"conic short 1.0",
"conic 2.6",
"conic short 2.6",
"conic 5.0",
"conic short 5.0"]
)
rename(mddQDF,["method","1","5","10","25","50","75","90","95","99"])

mddQDF2 = mddQDF[[1,3,5,7,9,11,13,15,17,2,4,6,8,10,12,14,16,18],:]
mddQDF2 = rename(mddQDF2,["method","1","5","10","25","50","75","90","95","99"])





herfindahlMatrix = zeros(9,sampleSize) 
for i = 1:sampleSize
    herfindahlMatrix[:,i] = data2herfindahl(dataList[i])
end
herfindahlMatrix

herfindahlQuantiles = zeros(9,9)
for i = 1:9
    herfindahlQuantiles[i,:] = quantile(herfindahlMatrix[i,:],[1,5,10,25,50,75,90,95,99]/100)
end

herfindahlQuantiles


sixtydayindices = setdiff(1:sampleSize,nonWorkingIndices)
sixtydayreturns = zeros(18,sampleSize)
for i = sixtydayindices
    sixtydayreturns[:,i] = ( data2quarterlyreturn(dataList[i],250) .+1 ) *100
end

sixtydayreturns = sixtydayreturns[:,sixtydayindices]

returnQuantiles = zeros(18,9)
for i = 1:18
    returnQuantiles[i,:] =quantile(sixtydayreturns[i,:],[1,5,10,25,50,75,90,95,99]/100)
end

rQDF = DataFrame(returnQuantiles,:auto)
insertcols!(rQDF,1,:method=>["w_optimMPT",
"w_optimMPT_short",
"w_optimcvar95",
"w_optimcvar95_short",
"w_optimcvar99",
"w_optimcvar99_short",
"w_optimvar",
"w_optimvar_short",
"conic 0.14",
"conic short 0.14",
"conic 0.4",
"conic short 0.4",
"conic 1.0",
"conic short 1.0",
"conic 2.6",
"conic short 2.6",
"conic 5.0",
"conic short 5.0"]
)
rename(rQDF,["method","1","5","10","25","50","75","90","95","99"])

rQDF2 = rQDF[[1,3,5,7,9,11,13,15,17,2,4,6,8,10,12,14,16,18],:]
rQDF2 = rename(rQDF2,["method","1","5","10","25","50","75","90","95","99"])

