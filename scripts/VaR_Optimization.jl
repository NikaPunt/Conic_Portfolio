include("HeaderFile.jl")

println("Importing datasets")
begin
    names = ["BRK-B","NKE","FB","V","GOOGL"];
    csv_brk = CSV.File("datasets/stocks/BRK-B.csv");
    csv_nke = CSV.File("datasets/stocks/NKE.csv");
    csv_fb = CSV.File("datasets/stocks/FB.csv");
    csv_v = CSV.File("datasets/stocks/V.csv");
    csv_googl = CSV.File("datasets/stocks/GOOGL.csv");

    df_brk = DataFrame(csv_brk) # NIKE(NKE) GOOGLE(GOOGL) VISA(V) META(FB) BERKSHIRE(BRK-A)
    df_nke = DataFrame(csv_nke)
    df_fb = DataFrame(csv_fb)
    df_v = DataFrame(csv_v)
    df_googl = DataFrame(csv_googl)

    # list of all stock dataframes.
    df_list = [df_brk, df_nke, df_fb, df_v, df_googl];

    # list of all stock dividend yields.
    yields = [0, 0.0078, 0, 0.0069, 0];
end

println("Calculating daily returns")
begin
    nrAssets = length(df_list); #number of assets
    df_adj_cl_length = length(df_list[1]."Adj Close")
    sampleRtrns = zeros(nrAssets,df_adj_cl_length-1) #matrix containing the daily returns of each asset in each row.
    assetShiftedRtrns = Array{Float64,2}(undef,nrAssets,df_adj_cl_length-1); #same as up here but then making mean = 0

    for i = 1:nrAssets
        df = df_list[i]
        closes = df."Adj Close"
        rtrns = log.(closes[2:end]./closes[1:end-1])
        sampleRtrns[i,:] = rtrns
        gemiddelde = mean(rtrns)
        shiftedRtrns = rtrns-repeat([gemiddelde],length(rtrns))
        assetShiftedRtrns[i,:] = shiftedRtrns;
    end
end

println("Starting Independent Component Analysis")
begin
    ica_comps = ICA_assets(assetShiftedRtrns);
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
    V = VG_Params2MRet([getVGParams(pars[1,:]...,1,20),
                    getVGParams(pars[2,:]...,1,20),
                    getVGParams(pars[3,:]...,1,20),
                    getVGParams(pars[4,:]...,1,20),
                    getVGParams(pars[5,:]...,1,20)], M)
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
