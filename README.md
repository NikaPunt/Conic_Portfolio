This was my master's thesis on conic portfolio theory. 
My project was a deep analysis into the regulatory fitness of conic financial theory with respect to portfolio selection.

The code in this repository was built to calculate four different portfolios:
1. Markowitz minimum variance portfolio
2. Value-at-Risk (VaR) portfolio
3. Tail VaR (also named conditional VaR) portfolio
4. Conic portfolio

The code for 1 and 3 are not based on prior distributional assumptions. We can calculate those directly on the return data of a portfolio.
The code for 2 uses a heuristic outlined in [1]. 
The code for 4 is based on the algorithm outlined in [2].


![Conic Minimum Risk Frontier](https://i.imgur.com/3V5xtDD.png)




! Disclaimer !
knitro artelys was used to perform the optimization of the nonlinear optimization problem posed by the conic portfolio optimization. 



[1] Larsen, N., Mausser, H., and Uryasev, S. (2002). Algorithms for optimization of value-at-risk. In Financial engineering, E-commerce and supply chain, pages 19–46. Springer.

[2] Madan, D. B. (2016). Conic portfolio theory. International Journal of Theoretical and Applied
Finance, 19(03):1650019
