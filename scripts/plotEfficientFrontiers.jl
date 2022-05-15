include("HeaderFile.jl")

# struct Asset
#     name::String    
#     rtrns::Vector{Float64}
#     mean::Float64
#     vol::Float64
# end

println("Importing datasets")
begin
    filenames = readdir("data/DJ2020")
    N = length(filenames)
    uniqueNames = Vector{String}(undef,N)
    for i = 1:N
        name = filenames[i]
        coin = name[1:end-15]
        uniqueNames[i] = coin
    end



    df_list = Vector{DataFrame}(undef,N)
    for i = 1:N
        df_list[i] = DataFrame(CSV.File("data/DJ2020/"*filenames[i],delim=","))
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
y2k_pos = findall(x->x==Date(2020,02,14),df_list[1]."Date")[1] #position 896

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
        Rᵢ = log.(closes[2:end]./closes[1:end-1])
        beforeCrashRtrns = Rᵢ[1:y2k_pos]
        afterCrashRtrns = Rᵢ[y2k_pos+1:end]
        sampleRtrns[i,:] = beforeCrashRtrns
        sampleCrashRtrns[i,:] = afterCrashRtrns
        gemiddelde = 30.437*mean(beforeCrashRtrns)
        shiftedRtrns = beforeCrashRtrns-repeat([gemiddelde],length(beforeCrashRtrns))
        assetShiftedRtrns[i,:] = shiftedRtrns
        volatiliteit = sqrt(30.437)*std(beforeCrashRtrns)
        AssetArray[i] = Asset(name,beforeCrashRtrns,gemiddelde,volatiliteit)
    end
end

w_optimMPT = getMinVolWeights(AssetArray,false)
w_optimMPT_short = getMinVolWeights(AssetArray,true)


println("Calculating Efficient MPT Frontier")
begin
    # let us discretize a bunch of values for μ_required
    μ = means(AssetArray);
    Σ = 30.437*cov(hcat(rtrns(AssetArray)...));
    μ_required = range(minimum(μ),stop=maximum(μ),length=100);
    σ_efficient = zeros(100);

    model_frontier = Model();
    set_optimizer(model_frontier, Ipopt.Optimizer)
    set_optimizer_attribute(model_frontier, "constr_viol_tol", 1e-15)
    set_optimizer_attribute(model_frontier, "acceptable_tol", 1e-15)
    set_optimizer_attribute(model_frontier, "print_level", 0)
    @variable(model_frontier, w[1:nrAssets] >= 0) # you can unregister w through unregister(model_frontier, w)
    @constraint(model_frontier, lt1, sum(w) == 1);
    @objective(model_frontier, Min, w'*Σ*w);

    for i = 1:100
        println(i);
        μ_i = μ_required[i];
        @constraint(model_frontier, con, transpose(w)*μ == μ_i);
        optimize!(model_frontier);
        σ_efficient[i] = objective_value(model_frontier);
        delete(model_frontier,con);
        unregister(model_frontier,:con);
    end
end


w_optimvar = vec(readdlm("data/w_optimvar2020.txt",Float64))
w_optimcvar95 = vec(readdlm("data/w_optimcvar95-2020.txt",Float64))
w_optimcvar99 = vec(readdlm("data/w_optimcvar99-2020.txt",Float64))

weights = [w_optimvar,w_optimcvar95,w_optimcvar99,five_γ...]
μ_weights = [a'*μ for a in weights]
σ_weights = [w'*Σ*w for w in weights]

μ_minimum = w_optimMPT'*μ
σ_minimum = w_optimMPT'*Σ*w_optimMPT

MPT_efficientfrontier = PlotlyJS.scatter(;x=σ_efficient,y=μ_required,name="Minimum Variance Frontier")
MPT_dot = PlotlyJS.scatter(;x=[σ_minimum],y=[μ_minimum],mode="markers+text",text=["MinVar"],textposition="middle right", name="MinVar")
one =   PlotlyJS.scatter(;  x=[σ_weights[1]],y=[μ_weights[1]],mode="markers",name="VaR95")
two =   PlotlyJS.scatter(;  x=[σ_weights[2]],y=[μ_weights[2]],mode="markers",name="CVaR95")
three = PlotlyJS.scatter(;  x=[σ_weights[3]],y=[μ_weights[3]],mode="markers",name="CVaR99")
four =  PlotlyJS.scatter(;  x=[σ_weights[4]],y=[μ_weights[4]],mode="markers",name="Conic γ=$(a[1])")
five =  PlotlyJS.scatter(;  x=[σ_weights[5]],y=[μ_weights[5]],mode="markers",name="Conic γ=$(a[2])")
six =   PlotlyJS.scatter(;  x=[σ_weights[6]],y=[μ_weights[6]],mode="markers",name="Conic γ=$(a[3])")
seven = PlotlyJS.scatter(;  x=[σ_weights[7]],y=[μ_weights[7]],mode="markers",name="Conic γ=$(a[4])")
eight = PlotlyJS.scatter(;  x=[σ_weights[8]],y=[μ_weights[8]],mode="markers",name="Conic γ=$(a[5])")
nine =  PlotlyJS.scatter(;  x=[σ_weights[9]],y=[μ_weights[9]],mode="markers",name="Conic γ=$(a[6])")

PlotlyJS.plot([MPT_efficientfrontier,MPT_dot,one,two,three,four,five,six,seven,eight,nine],
            Layout(;title="Minimum Variance Frontier",
                    xaxis=attr(title="Volatility (Dec.)"),
                    yaxis=attr(title="Yield per month")))


a = vec([0.001 0.14 0.4 1.0 2.6 7.0])
five_γ = Vector{Vector{Float64}}(undef,6)
for i = 1:6
    nummer = a[i] 
    five_γ[i] = vec(readdlm("data/w_optimgap$nummer-2020.txt",Float64))
end
