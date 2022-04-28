include("HeaderFile.jl")

println("Importing datasets")
begin
    uniqueNames = ["FISI","GPS","JEQ","NLY","NVR","OCUL","PKE","RY","SMFG","TTGT"];

    df_fisi = DataFrame(CSV.File("datasets/randomstocks10/FISI.csv"))
    df_gps  = DataFrame(CSV.File("datasets/randomstocks10/GPS.csv"))
    df_jeq  = DataFrame(CSV.File("datasets/randomstocks10/JEQ.csv"))
    df_nly  = DataFrame(CSV.File("datasets/randomstocks10/NLY.csv"))
    df_nvr  = DataFrame(CSV.File("datasets/randomstocks10/NVR.csv"))
    df_ocul = DataFrame(CSV.File("datasets/randomstocks10/OCUL.csv"))
    df_pke  = DataFrame(CSV.File("datasets/randomstocks10/PKE.csv"))
    df_ry   = DataFrame(CSV.File("datasets/randomstocks10/RY.csv"))
    df_smfg = DataFrame(CSV.File("datasets/randomstocks10/SMFG.csv"))
    df_ttgt = DataFrame(CSV.File("datasets/randomstocks10/TTGT.csv"))

    # list of all stock dataframes.
    df_list = [df_fisi,df_gps,df_jeq,df_nly,df_nvr,df_ocul,df_pke,df_ry,df_smfg,df_ttgt];
end

println("Calculating daily returns")
begin
    nrAssets = length(df_list); #number of assets
    df_adj_cl_length = length(df_list[1]."Adj Close")
    sampleReturns = zeros(nrAssets,df_adj_cl_length-1) #matrix containing the daily returns of each asset in each row.
    assetShiftedReturns = Array{Float64,2}(undef,nrAssets,df_adj_cl_length-1); #same as up here but then making mean = 0

    for i = 1:nrAssets
        df = df_list[i]
        closes = df."Adj Close"
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
    V = VG_Params2MRet([getVGParams(pars[i,:]...,1,20) for i = 1:nrAssets], M)
end

println("Transforming independent returns to joint returns")
begin
    #returns Y = A*V where A is the mixing matrix.
    Y = A*V;
    const N = size(Y,1);
    Returns = zeros(N,M);
    # Let us center the returns in Y 
    for i = 1:N
        avgSumExp = 1/M*sum(exp.(Y[i,:]))
        # println(avgSumExp)
        for m = 1:M
            Returns[i,m] = exp(Y[i,m])-avgSumExp
        end
    end
end

include("calcMinVaR.jl")
println("Starting VaR Optimization")
begin
    β = 0.95
    (w_optimvar, VaR) = calcMinVar(Returns,β,false)
    println("Best $β-VaR-Optimizing Weights: $w_optimvar \nWith VaR: $VaR")
end

println("Starting VaR Long-Short Optimization")
begin
    β = 0.95
    (w_optimvar_short, VaR_short) = calcMinVar(Returns,β,false,true)
    println("Best $β-VaR-Optimizing Weights: $w_optimvar_short \nWith VaR: $VaR_short")
end
