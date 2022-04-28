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

println("Starting CVaR Optimization")
begin
    β = 0.95
    q = size(Returns,2)
    mincvar_model = Model();
    # set_optimizer(mincvar_model, Ipopt.Optimizer)
    # set_optimizer_attribute(mincvar_model, "constr_viol_tol", 1e-15)
    # set_optimizer_attribute(mincvar_model, "acceptable_tol", 1e-15)
    set_optimizer(mincvar_model, Clp.Optimizer) #No primal feasability
    set_optimizer_attribute(mincvar_model, "LogLevel", 0)
    set_optimizer_attribute(mincvar_model, "Algorithm", 4)
    @variable(mincvar_model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
    @variable(mincvar_model, τ)
    @variable(mincvar_model, u[1:q] >= 0)
    @objective(mincvar_model, Min, τ+sum(u)/(1-β));
    @constraint(mincvar_model, con2, u' .>= -(w'*Returns .+ τ)/q)
    @constraint(mincvar_model, con, ones(nrAssets)'*w == 1);
    optimize!(mincvar_model);
    w_optimcvar = value.(w)
    optimcvar = objective_value(mincvar_model)
    delete(mincvar_model,con);
    delete.(mincvar_model,con2);
    unregister(mincvar_model,:con);
    unregister(mincvar_model,:con2);
    println("Best $β-CVaR-Optimizing Weights: $w_optimcvar \nWith CVaR: $optimcvar")
end

println("Calculating Efficient CVaR-frontier")
begin
    A = [assetMeans'; ones(nrAssets)']
    lengte = 100
    # b = [range(minimum(assetMeans),stop=maximum(assetMeans),length=lengte)';ones(lengte)']
    b = [range(minimum(assetMeans),stop=maximum(assetMeans),length=lengte)';ones(lengte)']
    ws = zeros(5,lengte)
    τs = zeros(lengte)

    cvar_efficient = zeros(lengte)
    β = 0.95
    q = size(Returns,2)
    model = Model();
    set_optimizer(model, Ipopt.Optimizer)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-15)
    set_optimizer_attribute(model, "acceptable_tol", 1e-15)
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
    @variable(model, τ)
    @variable(model, u[1:q] >= 0)
    @objective(model, Min, τ+sum(u)/(1-β));
    @constraint(model, con2, u' .>= -(w'*Returns .+ τ)/q)
    for i = 1:lengte
        # println("Thread number ",Threads.threadid()," working on iteration ",i,"/",lengte)
        println(i);
        @constraint(model, con, A*w .== b[:,i]);
        optimize!(model);
        ws[:,i] = value.(w)
        τs[i] = value.(τ)
        cvar_efficient[i] = objective_value(model);
        delete(model,con);
        unregister(model,:con);
    end
end
plot(cvar_efficient, b[1,:])


println("Starting CVaR Long-Short Optimization")
begin
    β = 0.95
    q = size(Returns,2)
    mincvar_model = Model();
    # set_optimizer(mincvar_model, Ipopt.Optimizer)
    # set_optimizer_attribute(mincvar_model, "constr_viol_tol", 1e-15)
    # set_optimizer_attribute(mincvar_model, "acceptable_tol", 1e-15)
    set_optimizer(mincvar_model, Clp.Optimizer) #No primal feasability
    set_optimizer_attribute(mincvar_model, "LogLevel", 0)
    set_optimizer_attribute(mincvar_model, "Algorithm", 4)
    @variable(mincvar_model, w[1:nrAssets]) # you can unregister w through unregister(model, w)
    @variable(mincvar_model, τ)
    @variable(mincvar_model, u[1:q] >= 0)
    @objective(mincvar_model, Min, τ+sum(u)/(1-β));
    @constraint(mincvar_model, con2, u' .>= -(w'*Returns .+ τ)/q)
    @constraint(mincvar_model, con, ones(nrAssets)'*w == 1);
    optimize!(mincvar_model);
    w_optimcvar_short = value.(w)
    optimcvar_short = objective_value(mincvar_model)
    delete(mincvar_model,con);
    delete.(mincvar_model,con2);
    unregister(mincvar_model,:con);
    unregister(mincvar_model,:con2);
    println("Best $β-CVaR-Optimizing Weights: $w_optimcvar_short \nWith CVaR: $optimcvar_short")
end