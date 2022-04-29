const timeFactor = 30.437 # 30.437 converts daily to monthly returns for crypto, take ~20 for stocks.

## MPT (mean-variance optimization)
struct Asset
    name::String    
    rtrns::Vector{Float64}
    mean::Float64
    vol::Float64
end

function getMinVolWeights(AssetArray::Array{Asset},short::Bool=false)::Vector{Float64}
    means(a::Vector{Asset}) = Vector{Float64}([ass.mean for ass in a])
    vols(a::Vector{Asset}) = Vector{Float64}([ass.vol for ass in a])
    rtrns(a::Vector{Asset}) = Vector{Vector{Float64}}([ass.rtrns for ass in a])
    
    #Make sure to run this at least once
    println("Calculating Optimal Mean-Variance Portfolio")
    begin
        Σ = timeFactor*cov(hcat(rtrns(AssetArray)...));
        model = Model();
        set_optimizer(model, Ipopt.Optimizer)
        set_optimizer_attribute(model, "constr_viol_tol", 1e-15)
        set_optimizer_attribute(model, "acceptable_tol", 1e-15)
        if (short==false)
            @variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
        else
            @variable(model, w[1:nrAssets]) # you can unregister w through unregister(model, w)
        end
        @constraint(model, lt1, sum(w) == 1);
        @objective(model, Min, w'*Σ*w);
        optimize!(model);
        w_optimMPT = value.(w)
        σ_minimum = objective_value(model)
        println("Best Mean-Variance Weights: $w_optimMPT \nWith Variance: $σ_minimum")
    end
    return w_optimMPT
end

## mean-CVaR optimization

function getMinCVaRWeights(Rtrns::Array{Float64,2},β::Float64=0.95, short::Bool=false)::Vector{Float64}
    q = size(Rtrns,2)
    mincvar_model = Model();
    set_optimizer(mincvar_model, Ipopt.Optimizer)
    set_optimizer_attribute(mincvar_model, "constr_viol_tol", 1e-15)
    set_optimizer_attribute(mincvar_model, "acceptable_tol", 1e-15)
    set_optimizer_attribute(mincvar_model, "print_level", 5)
    # set_optimizer(mincvar_model, Clp.Optimizer) #No primal feasability
    # set_optimizer_attribute(mincvar_model, "LogLevel", 0)
    # set_optimizer_attribute(mincvar_model, "Algorithm", 4)
    if (short==false)
        @variable(mincvar_model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
    else
        @variable(mincvar_model, w[1:nrAssets])
    end
    @variable(mincvar_model, τ)
    @variable(mincvar_model, u[1:q] >= 0)
    @objective(mincvar_model, Min, τ+sum(u)/(1-β));
    @constraint(mincvar_model, con2, u' .>= -(w'*Rtrns .+ τ)/q)
    @constraint(mincvar_model, con, ones(nrAssets)'*w == 1);
    optimize!(mincvar_model);
    w_optimcvar = value.(w)
    optimcvar = objective_value(mincvar_model)
    delete(mincvar_model,con);
    delete.(mincvar_model,con2);
    unregister(mincvar_model,:con);
    unregister(mincvar_model,:con2);
    println("Best $β-CVaR-Optimizing Weights: $w_optimcvar \nWith CVaR: $optimcvar")
    return w_optimcvar
end

## mean-VaR optimization
include("calcMinVaR.jl")

function getMinVaRWeights(Rtrns::Array{Float64,2},β::Float64=0.95,short::Bool=false)::Vector{Float64}    
    println("Starting VaR Optimization")
    (w_optimvar, VaR) = calcMinVar(Rtrns,β,true,short)
    println("Best $β-VaR-Optimizing Weights: $w_optimvar \nWith VaR: $VaR")
    return w_optimvar
end

## CPT (mean-conic-gap optimization)
function getMinConicWeights(Rtrns::Matrix{Float64},γ::Float64=0.1,short_bool::Bool=false)::Vector{Float64}
    N = size(Rtrns,1)
    M = size(Rtrns,2)
    Ψs = [MAXMINVAR(m/M,γ) for m=0:M]
    ΨminusΨ = [Ψs[m]-Ψs[m-1] for m=2:(M+1)]
    # RtrnsSorted = vcat([sort(Rtrns[i,:])' for i = 1:N]...)
    # println("Starting optimization with γ == $γ")
    # # optimization

    # println("\n---------------------------------------------------------\n")
    Σ_indcomp_1m = cov(Rtrns') #(T = 1 month) Covariance on the simulated returns
    Σ_cholupper = cholesky(Σ_indcomp_1m).U
    # Σ_sample_1d = cov(sampleRtrns') #(T= 1 day) Covariance on the sample returns
    Vols = zeros(1000000)
    for i = 1:length(Vols)
        weights = (rand(N).-0.5)
        weights = weights/sum(weights)
        Vols[i] = sqrt(weights'*Σ_indcomp_1m*weights)
    end
    Q = quantile(Vols*√(12),0.5)

    # Create a new Knitro solver instance
    kc = KNITRO.KN_new()

    # Objective function
    function callbackEvalF(kc, cb, evalRequest, evalResult, userParams)
        x = evalRequest.x

        evalResult.obj[1] = GAP(x,Rtrns,ΨminusΨ)
    
        return 0
    end

    # Add vars and bounds
    KNITRO.KN_add_vars(kc,N)
    if (short_bool==true)
        KNITRO.KN_set_var_lobnds(kc,repeat([-KNITRO.KN_INFINITY],N))
    else
        KNITRO.KN_set_var_lobnds(kc,(repeat([0.0],N)))
    end
    KNITRO.KN_set_var_upbnds(kc,repeat([KNITRO.KN_INFINITY],N))

    # Set Initial point (optional)
    #KNITRO.KN_set_var_primal_init_values(kc,w_init)
    
    # Add constraints and their bounds
    KNITRO.KN_add_cons(kc,2)
    KNITRO.KN_set_con_lobnds(kc,[1.,-KNITRO.KN_INFINITY])
    KNITRO.KN_set_con_upbnds(kc,[1.,Q/√(12)])

    # Sum of weights must equal 1.0
    indexVars0 = Cint[0:N-1...]
    coefs0 = repeat([1.0],N)
    KNITRO.KN_add_con_linear_struct(kc, 0, indexVars0, coefs0)

    # The following is just setting the constraint of the volatility cap
    dimA = N;   # A = [1, 0, 0, 0; 0, 0, 2, 0] has two rows */
    nnzA = N*(N+1) ÷ 2;
    indexRowsA = vcat([0:i for i = 0:N-1]...)
    indexVarsA = vcat([repeat([i],i+1) for i = 0:N-1]...)
    coefsA = vec(vec_triu_loop(Σ_cholupper))
    b = repeat([0.],N)
    KNITRO.KN_add_con_L2norm(kc, 1, dimA, nnzA, Cint[indexRowsA...], Cint[indexVarsA...], coefsA, b)

    cb = KNITRO.KN_add_objective_callback(kc, callbackEvalF)

    # Set minimize or maximize(if not set, assumed minimize)
    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MAXIMIZE)

    # nThreads = Sys.CPU_THREADS
    # if nThreads > 1
    #     println("Running Knitro Tuner in parallel with $nThreads threads.")
    #     KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_BLAS_NUMTHREADS, nThreads)
    #     KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_NUMTHREADS, nThreads)
    #     # KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_PAR_MSNUMTHREADS, nThreads)
    # end

    KNITRO.KN_set_param(kc,KNITRO.KN_PARAM_OUTLEV,0)

    nStatus = KNITRO.KN_solve(kc)

    println("Knitro converged with final status = ", nStatus)

    #** An example of obtaining solution information. */
    nStatus, objSol, x, _ = KNITRO.KN_get_solution(kc)
    println("  optimal objective value  = ", objSol)
    println("  optimal primal values x  = ", x)
    
    feasError = KNITRO.KN_get_abs_feas_error(kc)
    println("  feasibility violation    = ", feasError)
    optError = KNITRO.KN_get_abs_opt_error(kc)
    println("  KKT optimality violation = ", optError)
    
    #** Delete the Knitro solver instance. */
    KNITRO.KN_free(kc)

    w_optim = x

    println("Best weights: $w_optim \nWith gap: $objSol")
    println("\n---------------------------------------------------------\n")
    return w_optim
end

function simulateJointReturns(assetShiftedRtrns::Array{Float64,2},M::Int=10000)::Array{Float64,2}
    println("Starting Independent Component Analysis")
    begin
        ica_comps = ICA_assets(assetShiftedRtrns);
        # X = A*S      where X are the returns, A is mixing matrix and S is independent comps
        # S = W'*X
        # W = ica_comps.W;
        A = ica_comps.mixing;
        S = ica_comps.indcomps; #these are the independent components. For each row, we need σ, ν, θ. 
    end

    println("Extracting implied VG parameters on independent components")
    pars = get_params_timeseries_returns(S);

    println("Simulating ",M," monthly returns of independent components")
    begin
        V = VG_Params2MRet([getVGParams(pars[i,:]...,1,timeFactor) for i = 1:nrAssets], M)
    end

    println("Transforming independent returns to joint returns")
    begin
        #returns Y = A*V where A is the mixing matrix.
        Y = A*V;
        N = size(Y,1);
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
    return Rtrns
end