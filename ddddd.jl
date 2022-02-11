model = Model();
set_optimizer(model, Ipopt.Optimizer)
@variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
@constraint(model, lt1, sum(w) == 1);
@objective(model, Min, w'*Σ*w);
optimize!(model);

w_optimMPT = value.(w)


μ_i = μ_required[70];
@constraint(model, con, transpose(w)*μ == μ_i);
optimize!(model);
# σ_efficient[i] = objective_value(model);
delete(model,con);
unregister(model,:con);
w_optimSharpe = value.(w)

