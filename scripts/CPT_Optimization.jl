include("MPT_Optimization.jl")
function CPT_Optimization(Rtrns::Matrix{Float64},γ::Float64=0.1,(short_bool,short_weights)::Tuple{Bool,Vector{Float64}}=(false,[0.0]))
    N = size(Rtrns,1)
    M = size(Rtrns,2)
    Ψs = [MAXMINVAR(m/M,γ) for m=0:M]
    ΨminusΨ = [Ψs[m]-Ψs[m-1] for m=2:(M+1)]
    RtrnsSorted = vcat([sort(Rtrns[i,:])' for i = 1:N]...)
    println("Starting optimization with γ == $γ")
    # # optimization
    w_optim = zeros(N)
    gap_optim = 0
    # The following optimizes for the best gap.
    if short_bool == false
        println("\n---------------------------------------------------------\n")
        println("Optimize long-only weights for gaps -")
        x_0 = repeat([1/N], N);
        lx = zeros(N); ux = ones(N);
        lc = [0]; uc = [0];
        fun1(x) = -GAPSORTED(x,Rtrns,RtrnsSorted,ΨminusΨ)
        df = TwiceDifferentiable(fun1, x_0)
        con_c1!(c, x) = (c[1] = sum(x)-1; c)
        dfc = TwiceDifferentiableConstraints(con_c1!, lx, ux, lc, uc)
        res = optimize(df, dfc, x_0, IPNewton())
        w_optim = Optim.minimizer(res)
        gap_optim = -Optim.minimum(res)
        println("Best weights: $w_optim \nWith gap: $gap_optim")
        return (w_optim,gap_optim)
    end
    println("\n---------------------------------------------------------\n")
    println("Optimize long-short weights for gaps -")
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
    KNITRO.KN_set_var_lobnds(kc,repeat([-KNITRO.KN_INFINITY],N))
    # x_L = repeat([-KNITRO.KN_INFINITY],N)
    KNITRO.KN_set_var_upbnds(kc,repeat([KNITRO.KN_INFINITY],N))
    # x_U = repeat([KNITRO.KN_INFINITY],N)

    # Set Initial point (optional)
    #KNITRO.KN_set_var_primal_init_values(kc,short_weights)
    
    # Add constraints and their bounds
    KNITRO.KN_add_cons(kc,2)
    # m = 2
    # c_Type = [KTR_CONTYPE_LINEAR]
    KNITRO.KN_set_con_lobnds(kc,[1.,-KNITRO.KN_INFINITY])
    # c_L = [-KTR_INFBOUND]
    KNITRO.KN_set_con_upbnds(kc,[1.,Q/√(12)])
    # c_U = [3.0]

    indexVars0 = Cint[0:N-1...]
    coefs0 = repeat([1.0],N)
    KNITRO.KN_add_con_linear_struct(kc, 0, indexVars0, coefs0)

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

    # fun2(x) = -GAP(x,Rtrns,ΨminusΨ)
    # x_0 = short_weights
    # df = TwiceDifferentiable(fun2, x_0)
    # con_c2!(c, x) = (c[1] = sum(x)-1; c[2] = sqrt(x'*Σ_indcomp_1m*x); c)
    # lx = fill(-Inf,N); ux = fill(Inf,N);
    # lc = [0,0]; uc = [0,Q/√(12)];
    # dfc = TwiceDifferentiableConstraints(con_c2!, lx, ux, lc, uc)
    # res2 = optimize(df, dfc,x_0, IPNewton())
    # w_optim = Optim.minimizer(res2)
    # gap_optim  = GAP(w_optim,Rtrns,ΨminusΨ)
    # println("Best weights: $w_optim \nWith gap: $gap_optim")
    # println("\n---------------------------------------------------------\n")
    w_optim = x
    gap_optim = objSol
    println("Best weights: $w_optim \nWith gap: $gap_optim")
    println("\n---------------------------------------------------------\n")
    return (w_optim, gap_optim)
end

function vec_triu_loop(M::AbstractMatrix{T}) where T
    m, n = size(M)
    m == n || throw(error("not square"))
    l = n*(n+1) ÷ 2
    v = Vector{T}(undef,l)
    k = 0
    @inbounds for i in 1:n
        for j in 1:i
            v[k + j] = M[j, i]
        end
        k += i
    end
    v
end