using Distributions
N(x) = cdf.(Normal(), x)
N_inv(p) = quantile.(Normal(),p)
function Wang(u,λ)
    wang = N(N_inv(u)+λ)
    return wang
end