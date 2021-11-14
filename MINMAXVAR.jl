function MINMAXVAR(u,λ)
    minmaxvar = 1-(1-u^(1/(λ+1)))^(λ+1)
    return minmaxvar
end