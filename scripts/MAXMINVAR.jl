function MAXMINVAR(u,λ)
    maxminvar = exp((1/(λ+1))*log(1-exp((λ+1)*log((1-u)))))
    return maxminvar
end