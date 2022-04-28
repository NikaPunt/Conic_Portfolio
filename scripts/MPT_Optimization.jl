function MPT_Optimization(Rtrns::Matrix{Float64},Tnew::Float64=30.437) #convert to monthly
    Σ = Tnew*cov(Rtrns')
    model = Model();
    set_optimizer(model, Ipopt.Optimizer)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-15)
    set_optimizer_attribute(model, "acceptable_tol", 1e-15)
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model, w[1:nrAssets]) # you can unregister w through unregister(model, w)
    @constraint(model, lt1, sum(w) == 1);
    @objective(model, Min, w'*Σ*w);
    optimize!(model);
    w_optimMPT = value.(w)
    σ_minimum = objective_value(model)
    return (w_optimMPT, σ_minimum)
end