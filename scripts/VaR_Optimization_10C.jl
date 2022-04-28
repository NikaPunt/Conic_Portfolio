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
    df_adj_cl_length = length(df_list[1]."close")
    sampleReturns = zeros(nrAssets,df_adj_cl_length-1) #matrix containing the daily returns of each asset in each row.
    assetShiftedReturns = Array{Float64,2}(undef,nrAssets,df_adj_cl_length-1); #same as up here but then making mean = 0

    for i = 1:nrAssets
        df = df_list[i]
        closes = df."close"
        rtrns = log.(closes[2:end]./closes[1:end-1])
        sampleReturns[i,:] = rtrns
        gemiddelde = mean(rtrns)
        shiftedReturns = rtrns-repeat([gemiddelde],length(rtrns))
        assetShiftedReturns[i,:] = shiftedReturns;
    end
end

println("Starting Independent Component Analysis")
begin
    ica_comps = ICA_assets(assetShiftedReturns);
    # X = A*S      where X are the returns, A is mixing matrix and S is independent comps
    # S = W'*X
    W = ica_comps.W;
    A = ica_comps.mixing;
    S = ica_comps.indcomps; #these are the independent components. For each row, we need σ, ν, θ. 
end

println("Extracting implied VG parameters on independent components")
pars = get_params_timeseries_returns(S);

const M = 10000
println("Simulating ",M," monthly returns of independent components")
begin
    V = VG_Params2MRet([getVGParams(pars[i,:]...,1,30.437) for i = 1:nrAssets], M)
end

println("Transforming independent returns to joint returns")
begin
    #returns Y = A*V where A is the mixing matrix.
    Y = A*V;
    const N = size(Y,1);
    Rtrns = zeros(N,M);
    # Let us center the returns in Y 
    for i = 1:N
        avgSumExp = 1/M*sum(exp.(Y[i,:]))
        # println(avgSumExp)
        for m = 1:M
            Rtrns[i,m] = exp(Y[i,m])-avgSumExp
        end
    end
end

include("calcMinVaR.jl")
println("Starting VaR Optimization")
begin
    β = 0.95
    (w_optimvar, VaR) = calcMinVar(Rtrns,β,false)
    println("Best $β-VaR-Optimizing Weights: $w_optimvar \nWith VaR: $VaR")
end

println("Starting VaR Long-Short Optimization")
begin
    β = 0.95
    (w_optimvar_short, VaR_short) = calcMinVar(Rtrns,β,false,true)
    println("Best $β-VaR-Optimizing Weights: $w_optimvar_short \nWith VaR: $VaR_short")
end
