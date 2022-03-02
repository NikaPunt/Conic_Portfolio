##ICA_assets
# The function ICA_assets(assets) calculates the independent components on the returns of the assets
# Make sure returns have mean zero.

# import Pkg;
# Pkg.add("IndependentComponentAnalysis")
using IndependentComponentAnalysis

struct ICA_components
    mixing::Matrix{Float64}
    W::Matrix{Float64}
    indcomps::Matrix{Float64}
end #returns = mixing*indcomps    and     indcomps = W'*returns

function ICA_assets(returns::Matrix{Float64})
    nrAssets = size(returns)[1]
    ica = IndependentComponentAnalysis.fit(ICA, returns, nrAssets, FastICA();
                fun       = IndependentComponentAnalysis.Tanh(),
                do_whiten = true,
                maxiter   = 10000,
                tol       = 1e-6,
                mean      = 0,
                winit     = nothing
      )
    S = IndependentComponentAnalysis.transform(ica,returns)
    return ICA_components(inv(ica.W'),ica.W
    ,S)
end