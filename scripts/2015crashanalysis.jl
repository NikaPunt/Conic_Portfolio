#Bring everything together

include("HeaderFile.jl")
println("Importing datasets")
using Dates
begin
    filenames = readdir("data/DJ2015")
    N = length(filenames)
    uniqueNames = Vector{String}(undef,N)
    for i = 1:N
        name = filenames[i]
        coin = name[1:end-15]
        uniqueNames[i] = coin
    end



    df_list = Vector{DataFrame}(undef,N)
    for i = 1:N
        df_list[i] = DataFrame(CSV.File("data/DJ2015/"*filenames[i],delim=","))
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

# let's find 2007-09-27 (DJIA goes down here)
y2k_pos = findall(x->x==Date(2015,03,02),df_list[1]."Date")[1] #position 896

# Extract DJIA from df_list
DJI_index = findall(x->x=="DJI",uniqueNames)[1]
DJI = df_list[DJI_index]
DJI_val = DJI."Close"
allbutdji = repeat([true],N); allbutdji[DJI_index]=false;allbutdji

df_list = df_list[allbutdji]
uniqueNames = uniqueNames[allbutdji]
println("Calculating daily returns")
begin
    nrAssets = length(df_list); #number of assets
    sampleRtrns = zeros(nrAssets,y2k_pos) #matrix containing the daily returns of each asset in each row.
    sampleCrashRtrns = zeros(nrAssets,length(df_list[1]."Close")-1-y2k_pos)
    sampleCrashValue = zeros(nrAssets,length(df_list[1]."Close")-y2k_pos)
    assetShiftedRtrns = Array{Float64,2}(undef,nrAssets,y2k_pos); #same as up here but then making mean = 0
    AssetArray = Array{Asset}(undef,nrAssets);
    for i = 1:nrAssets
        df = df_list[i]
        name = uniqueNames[i]
        closes = df."Close"
        sampleCrashValue[i,:] = closes[y2k_pos+1:end]
        rtrns = log.(closes[2:end]./closes[1:end-1])
        beforeCrashRtrns = rtrns[1:y2k_pos]
        afterCrashRtrns = rtrns[y2k_pos+1:end]
        sampleRtrns[i,:] = beforeCrashRtrns
        sampleCrashRtrns[i,:] = afterCrashRtrns
        gemiddelde = mean(beforeCrashRtrns)
        shiftedRtrns = beforeCrashRtrns-repeat([gemiddelde],length(beforeCrashRtrns))
        assetShiftedRtrns[i,:] = shiftedRtrns
        volatiliteit = sqrt(30.437)*std(beforeCrashRtrns)
        AssetArray[i] = Asset(name,beforeCrashRtrns,gemiddelde,volatiliteit)
    end
end

w_optimMPT = getMinVolWeights(AssetArray,false)
w_optimMPT_short = getMinVolWeights(AssetArray,true)

# Rtrns = simulateJointReturns(assetShiftedRtrns, 100000)
Rtrns = readdlm("data/Rtrns2015.txt",Float64)
using DelimitedFiles
open("data/Rtrns2015.txt", "w") do io
    writedlm(io, Rtrns)
end

# w_optimvar = getMinVaRWeights(Rtrns,0.95,false)
w_optimvar = vec(readdlm("data/w_optimvar2015.txt",Float64))
# w_optimvar_short = getMinVaRWeights(Rtrns,0.95,true)
w_optimvar_short = vec(readdlm("data/w_optimvar2015_short.txt",Float64))
# w_optimcvar95 = getMinCVaRWeights(Rtrns,0.95,false)
w_optimcvar95 = vec(readdlm("data/w_optimcvar95-2015.txt",Float64))
# w_optimcvar95_short = getMinCVaRWeights(Rtrns,0.95,true)
w_optimcvar95_short = vec(readdlm("data/w_optimcvar95-2015_short.txt",Float64))

# w_optimcvar99 = getMinCVaRWeights(Rtrns,0.99,false)
w_optimcvar99 = vec(readdlm("data/w_optimcvar99-2015.txt",Float64))
# w_optimcvar99_short = getMinCVaRWeights(Rtrns,0.99,true)
w_optimcvar99_short = vec(readdlm("data/w_optimcvar99-2015_short.txt",Float64))


open("data/w_optimvar2015.txt", "w") do file
    writedlm(file, w_optimvar)
end
open("data/w_optimvar2015_short.txt", "w") do file
    writedlm(file, (w_optimvar_short))
end
open("data/w_optimcvar95-2015.txt", "w") do file
    writedlm(file, (w_optimcvar95))
end
open("data/w_optimcvar95-2015_short.txt", "w") do file
    writedlm(file, (w_optimcvar95_short))
end
open("data/w_optimcvar99-2015.txt", "w") do file
    writedlm(file, (w_optimcvar99))
end
open("data/w_optimcvar99-2015_short.txt", "w") do file
    writedlm(file, (w_optimcvar99_short))
end

a = vec([0.001 0.14 0.4 1.0 2.6 7.0])
five_γ = Vector{Vector{Float64}}(undef,6)
five_γ_short = Vector{Vector{Float64}}(undef,6)
Threads.@threads for i = 1:6
    id = Threads.threadid()
    println("Thread $id on iteration $i")
    nummer = a[i] 
    five_γ[i] = getMinConicWeights(Rtrns,a[i],false)
    open("data/w_optimgap$nummer-2015.txt", "w") do file
        writedlm(file, (five_γ[i]))
    end 
    five_γ_short[i] = getMinConicWeights(Rtrns,a[i],true)
    open("data/w_optimgap$nummer-2015_short.txt", "w") do file
        writedlm(file, (five_γ_short[i]))
    end 
end
# for i = 1:6
#     nummer = a[i] 
#     five_γ[i] = vec(readdlm("data/w_optimgap$nummer-2015.txt",Float64))
#     five_γ_short[i] = vec(readdlm("data/w_optimgap$nummer-2015_short.txt",Float64))
# end

beginning = [1]

k = 1
allRtrns = hcat(sampleRtrns,sampleCrashRtrns);

begin
    indexChangeMPT = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimMPT
    indexMPT = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangeMPT[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangeMPT_short = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimMPT_short
    indexMPT_short = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangeMPT_short[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangevar = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimvar
    indexvar = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangevar[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangevar_short = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimvar_short
    indexvar_short = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangevar_short[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangecvar95 = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimcvar95
    indexcvar95 = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangecvar95[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangecvar95_short = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimcvar95_short
    indexcvar95_short = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangecvar95_short[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangecvar99 = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimcvar99
    indexcvar99 = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangecvar99[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangecvar99_short = 1 .+ allRtrns[:,y2k_pos+k:end]'*w_optimcvar99_short
    indexcvar99_short = DJI_val[y2k_pos+k] .*vcat(beginning, [prod(indexChangecvar99_short[1:i]) for i = 1:length(indexChangeMPT)])
    indexChangefive_γ = [1 .+ allRtrns[:,y2k_pos+k:end]'*five_γ[i] for i = 1:6]
    indexfive_γ = [1 1 1 1 1 1;zeros(length(indexChangeMPT),6)]
    for i = 1:6
        indexChangegap = indexChangefive_γ[i]
        indexfive_γ[2:end,i] = [prod(indexChangegap[1:i]) for i = 1:length(indexChangeMPT)]
    end
    indexfive_γ = DJI_val[y2k_pos+k] .*indexfive_γ
    indexChangefive_γ_short = [1 .+ allRtrns[:,y2k_pos+k:end]'*five_γ_short[i] for i = 1:6]
    indexfive_γ_short = [1 1 1 1 1 1;zeros(length(indexChangeMPT),6)]
    for i = 1:6
        indexChangegap = indexChangefive_γ_short[i]
        indexfive_γ_short[2:end,i] = [prod(indexChangegap[1:i]) for i = 1:length(indexChangeMPT)]
    end
    indexfive_γ_short = DJI_val[y2k_pos+k] .*indexfive_γ_short
end

using PlotlyJS
function plotonlyconic()
    crashdate = df_list[1]."Date"[y2k_pos+k:end]
    one =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,1], mode="lines",name = "Conic γ=$(a[1])")
    two =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,2], mode="lines",name = "Conic γ=$(a[2])")
    three = PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,3], mode="lines",name = "Conic γ=$(a[3])")
    four =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,4], mode="lines",name = "Conic γ=$(a[4])")
    five =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,5], mode="lines",name = "Conic γ=$(a[5])")
    six =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,5], mode="lines",name = "Conic γ=$(a[6])")
    ten = PlotlyJS.scatter(;x=crashdate, y=DJI_val[y2k_pos+k:end], mode="lines",name = "DJIA")
    PlotlyJS.plot([one,two,three,four,five,six,ten])
end
plotonlyconic()

function plotonlynonconic()
    crashdate = df_list[1]."Date"[y2k_pos+k:end]
    six =   PlotlyJS.scatter(;x=crashdate, y=indexMPT, mode="lines",name = "Mean-Variance")
    seven = PlotlyJS.scatter(;x=crashdate, y=indexvar, mode="lines",name = "Mean-VaR")
    eight = PlotlyJS.scatter(;x=crashdate, y=indexcvar95, mode="lines",name = "Mean-CVaR 95%")
    nine = PlotlyJS.scatter(;x=crashdate, y=indexcvar99, mode="lines",name = "Mean-CVaR 99%")
    ten = PlotlyJS.scatter(;x=crashdate, y=DJI_val[y2k_pos+k:end], mode="lines",name = "DJIA")
    PlotlyJS.plot([six,seven,eight,nine,ten])
end
plotonlynonconic()

function plotall()
    crashdate = df_list[1]."Date"[y2k_pos+k:end]
    one =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,1], mode="lines",name = "Conic γ=$(a[1])")
    two =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,2], mode="lines",name = "Conic γ=$(a[2])")
    three = PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,3], mode="lines",name = "Conic γ=$(a[3])")
    four =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,4], mode="lines",name = "Conic γ=$(a[4])")
    five =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,5], mode="lines",name = "Conic γ=$(a[5])")
    six =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ[:,5], mode="lines",name = "Conic γ=$(a[6])")
    seven =   PlotlyJS.scatter(;x=crashdate, y=indexMPT, mode="lines",name = "Mean-Variance")
    eight = PlotlyJS.scatter(;x=crashdate, y=indexvar, mode="lines",name = "Mean-VaR")
    nine = PlotlyJS.scatter(;x=crashdate, y=indexcvar95, mode="lines",name = "Mean-CVaR 95%")
    ten = PlotlyJS.scatter(;x=crashdate, y=indexcvar99, mode="lines",name = "Mean-CVaR 99%")
    eleven = PlotlyJS.scatter(;x=crashdate, y=DJI_val[y2k_pos+k:end], mode="lines",name = "DJIA",opacity=0.25)
    p = PlotlyJS.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven])
    if k == -365
        PlotlyJS.savefig(p, "plots/2015 crash/2015all-365.html")
    elseif k == -30
        PlotlyJS.savefig(p, "plots/2015 crash/2015all-30.html")
    else
        PlotlyJS.savefig(p, "plots/2015 crash/2015all.html")
    end
end
plotall()


function plotonlyconic_short()
    crashdate = df_list[1]."Date"[y2k_pos+k:end]
    one =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,1], mode="lines",name = "Conic γ=$(a[1])")
    two =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,2], mode="lines",name = "Conic γ=$(a[2])")
    three = PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,3], mode="lines",name = "Conic γ=$(a[3])")
    four =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,4], mode="lines",name = "Conic γ=$(a[4])")
    five =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,5], mode="lines",name = "Conic γ=$(a[5])")
    six =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,5], mode="lines",name = "Conic γ=$(a[6])")
    ten = PlotlyJS.scatter(;x=crashdate, y=DJI_val[y2k_pos+k:end], mode="lines",name = "DJIA")
    PlotlyJS.plot([one,two,three,four,five,six,ten])
end
plotonlyconic_short()

function plotonlynonconic_short()
    crashdate = df_list[1]."Date"[y2k_pos+k:end]
    six =   PlotlyJS.scatter(;x=crashdate, y=indexMPT_short, mode="lines",name = "Mean-Variance")
    seven = PlotlyJS.scatter(;x=crashdate, y=indexvar_short, mode="lines",name = "Mean-VaR")
    eight = PlotlyJS.scatter(;x=crashdate, y=indexcvar95_short, mode="lines",name = "Mean-CVaR 95%")
    nine = PlotlyJS.scatter(;x=crashdate, y=indexcvar99_short, mode="lines",name = "Mean-CVaR 99%")
    ten = PlotlyJS.scatter(;x=crashdate, y=DJI_val[y2k_pos+k:end], mode="lines",name = "DJIA")
    PlotlyJS.plot([six,seven,eight,nine,ten])
end
plotonlyconic_short()

function plotall_short()
    crashdate = df_list[1]."Date"[y2k_pos+k:end]
    one =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,1], mode="lines",name = "Conic γ=$(a[1])")
    two =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,2], mode="lines",name = "Conic γ=$(a[2])")
    three = PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,3], mode="lines",name = "Conic γ=$(a[3])")
    four =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,4], mode="lines",name = "Conic γ=$(a[4])")
    five =  PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,5], mode="lines",name = "Conic γ=$(a[5])")
    six =   PlotlyJS.scatter(;x=crashdate, y=indexfive_γ_short[:,5], mode="lines",name = "Conic γ=$(a[6])")
    seven =   PlotlyJS.scatter(;x=crashdate, y=indexMPT_short, mode="lines",name = "Mean-Variance")
    eight = PlotlyJS.scatter(;x=crashdate, y=indexvar_short, mode="lines",name = "Mean-VaR")
    nine = PlotlyJS.scatter(;x=crashdate, y=indexcvar95_short, mode="lines",name = "Mean-CVaR 95%")
    ten = PlotlyJS.scatter(;x=crashdate, y=indexcvar99_short, mode="lines",name = "Mean-CVaR 99%")
    eleven = PlotlyJS.scatter(;x=crashdate, y=DJI_val[y2k_pos+k:end], mode="lines",name = "DJIA",opacity=0.25)
    p = PlotlyJS.plot([one,two,three,four,five,six,seven,eight,nine,ten,eleven])
    if k == -365
        PlotlyJS.savefig(p, "plots/2015 crash/2015all-365_short.html")
    elseif k == -30
        PlotlyJS.savefig(p, "plots/2015 crash/2015all-30_short.html")
    else
        PlotlyJS.savefig(p, "plots/2015 crash/2015all_short.html")
    end
end
plotall_short()

a=vec(a)
yaxis = ["VaR 95%";"MPT";"γ=" .* string.(a);"CVaR 95%";"CVaR 99%"]
datamatrix = hcat(w_optimvar,w_optimMPT,five_γ...,w_optimcvar95,w_optimcvar99)'
PlotlyJS.plot(PlotlyJS.heatmap(y=yaxis,x=uniqueNames,z = datamatrix,title="Weights Long-Only"))

datamatrixshort = hcat(w_optimvar_short,w_optimMPT_short,five_γ_short...,w_optimcvar95_short,w_optimcvar99_short)'
PlotlyJS.plot(PlotlyJS.heatmap(y=yaxis,x=uniqueNames,z = datamatrixshort,title="Weights Long-Only"))
