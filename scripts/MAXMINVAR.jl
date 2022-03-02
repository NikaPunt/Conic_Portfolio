function MAXMINVAR(u,λ)
    maxminvar = (1-(1-u)^(λ+1))^(1/(λ+1))
    return maxminvar
end