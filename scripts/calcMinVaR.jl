# This code will implement the heuristic algorithm for finding
# a minimum VaR for a given portfolio.
#
# I.e. we implement alg A1 in:
# Larsen, Nicklas, Helmut Mausser, and Stanislav Uryasev. 
# "Algorithms for optimization of value-at-risk." 
# Financial engineering, E-commerce and supply chain. 
# Springer, Boston, MA, 2002. 19-46.
using JuMP # language
using AmplNLWriter # interface
using Ipopt # solver


#calcMinVar(returns::Matrix{Float64}, α::Float64=0.95) returns a Tuple(Vector{Float64},Float64) where
#position 1 is the weights and position 2 is the minimum value at risk. 
#
#Input:     returns
#           Type:           Matrix{Float64}
#           Description:    N×M matrix where each column is the joint return over a specific
#                           period of N different assets.
#           
#           α (optional)
#           Type:           Float64
#           Description:    A value between 0 and 1 to calculate the minimum α-VaR portfolio. 
#                           Default value is 0.95
#
#Output:    (weights,VaR)
#           Type:           Tuple(Vector{Float64},Float64)
#           Description:    The weights that form the minimum value-at-risk portfolio and the corresponding
#                           VaR value.
#
function calcMinVar(returns::Matrix{Float64},α::Float64=0.95,verbose::Bool=true,short::Bool=false)
    # step 0 - initialization
    i = 0
    αᵢ = α
    
    J = size(returns,2)
    H₀ = 1:J
    Hᵢ₋₁ = [1] # Doesn't matter what this is just yet
    Hᵢ = H₀
    ξ = 0.5
    nrAssets = size(returns,1)
    weights = zeros(nrAssets)
    VaR = 0


    # iteration
    while true
        Hcomp = setdiff(H₀,Hᵢ)
        ## step 1 - optimization subproblem
        ## step 1 (i)
        model = Model();
        # set_optimizer(model, Clp.Optimizer) #No primal feasability
        # set_optimizer_attribute(model, "LogLevel", 0)
        # set_optimizer_attribute(model, "Algorithm", 4)
        
        set_optimizer(model, Ipopt.Optimizer) # Memory runs out?!
        set_optimizer_attribute(model, "constr_viol_tol", 1e-15)
        set_optimizer_attribute(model, "acceptable_tol", 1e-15)
        set_optimizer_attribute(model, "print_level", 5)
        # set_optimizer(model, GLPK.Optimizer)
        # set_optimizer_attribute(model, "tm_lim", 60 * 1_000)
        # set_optimizer_attribute(model, "msg_lev", GLPK.GLP_MSG_OFF)

        if (short==false)
            @variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
        else
            @variable(model, w[1:nrAssets])
        end
        @variable(model, τ)
        @variable(model, γ >= 0)
        @variable(model, 0.0 <= u[1:length(Hᵢ)] <= 10000.)
        @objective(model, Min, τ + ones(length(Hᵢ))'*u/(J*(1-αᵢ)))
        @constraint(model, con_weight1, ones(nrAssets)'*w == 1.0)
        @constraint(model, con, u' .>= -(w'*returns[:,Hᵢ] .+ τ))
        @constraint(model, con_active, -w'*returns[:,Hᵢ] .<= γ)
        if (i > 0) @constraint(model, con_inactive, -w'*returns[:,Hcomp] .>= γ) end
        optimize!(model);

        ## step 1 (ii)
        weights = value.(w)
        perm = sortperm(vec(-weights'*returns))
        portfolio_losses_sorted = (-weights'*returns)[perm]
        delete.(model,con)
        delete.(model,con_active)
        if (i>0) delete.(model,con_inactive) end
        delete(model,u)
        unregister(model,:con)
        # unregister(model,:con_active)
        # if (i>0) unregister(model,:con_inactive) end
        unregister(model,:u)

        ## step 2 Estimating VaR
        l = minimum(H₀[H₀/J.>=α])
        VaR = portfolio_losses_sorted[l]

        ## step 3 Stop the algorithm
        if issetequal(Hᵢ, Hᵢ₋₁)
            # print(Hcomp, i)
            break
        # else
        #     println("Hᵢ₋₁\\Hᵢ: ", setdiff(Hᵢ₋₁,Hᵢ))
        #     println("Hᵢ\\Hᵢ₋₁: ", setdiff(Hᵢ,Hᵢ₋₁))
        end

        ## step 4 Reinitialization
        i = i + 1
        bᵢ = α + (1-α)*(1-ξ)^i
        αᵢ = α/bᵢ
        Hᵢ₋₁ = Hᵢ
        Hᵢ = intersect(perm[H₀/J .<= bᵢ],Hᵢ₋₁)

        if verbose
            println(weights)
            println(VaR)
        end
        ## step 5 algorithm modification
        # active_loss_max = portfolio_returns_sorted[minimum(H₀[H₀/J.>=bᵢ])]
        # inactive_losses = vec(weights'*returns[:,Hcomp])
        # nr_inactive_losses = length(inactive_losses)
        # Hᵢ[(end-(nr_inactive_losses-1)):end]
        # active_inactive_indices = Hcomp[inactive_losses .< active_loss_max]
        # Hᵢ = vcat(Hᵢ,active_inactive_indices)
        # println(length(active_inactive_indices))
    end
    return (weights, VaR)
end