# We will generate the box plots from the quantile data collected on Hedge fund performance: Sources and measures (Eberlein and Madan 2009)
# Based on                    .01     .05     .25     .5      .75     .95     .99
#           Stocks            0       0       .0205   .1177   .2444   .3243   .3525
# MINVAR    Indices           0       0       .0515   .1265   .1945   .2492   .2492
#           Funds             .0438   .1640   .4216   .7175   1.1345  2.0668  3.43
#           Stocks            0       0       .0188   .0883   .1983   .2779   .3097
# MAXVAR    Indices           0       0       .0423   .0957   .1564   .1964   .1964
#           Funds             .0407   .1374   .3309   .4966   .7142   1.2346  1.9026
#           Stocks            0       0       .0098   .0494   .1079   .1426   .1563
# MAXMINVAR Indices           0       0       .0231   .0535   .0863   .1062   .1062
#           Funds             .0214   .0738   .1760   .2679   .3892   .6591   .9670
#           Stocks            0       0       .0098   .0488   .1049   .1372   .1499
# MINMAXVAR Indices           0       0       .0229   .0527   .0844   .1032   .1032
#           Funds             .0212   .0726   .1673   .2495   .3529   .5645   .7887

# import Pkg; Pkg.add("PlotlyJS")
using PlotlyJS

MINVAR_qt = box(y=[.0438,   .1640,   .4216,   .4216,   .4216,   .4216,   .7175,   1.1345,  1.1345,  1.1345,  1.1345,  2.0668,  3.43], name="MINVAR")
MAXVAR_qt = box(y=[.0407,   .1374,   .3309,   .3309,   .3309,   .3309,   .4966,   .7142,   .7142,   .7142,   .7142,   1.2346,  1.9026], name="MAXVAR")
MAXMINVAR_qt = box(y=[.0214,   .0738,   .1760,   .1760,   .1760,   .1760,   .2679,   .3892,   .3892,   .3892,   .3892,  .6591,   .9670], name="MAXMINVAR")
MINMAXVAR_qt = box(y=[.0212,   .0726,   .1673,   .1673,   .1673,   .1673,   .2495,   .3529,   .3529,   .3529,   .3529,   .5645,   .7887], name="MINMAXVAR")


Data = [MINVAR_qt, MAXVAR_qt, MAXMINVAR_qt, MINMAXVAR_qt]

plot(Data)