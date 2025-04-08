import logging
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from cav.utils import util, config
from portSim.utils import portSimUtil as psutil
from portSim.simulation.Parameter import FloatParameter, IntParameter
from portSim.simulation.ListParameter import ListParameter
from portSim.simulation.VectorParameter import FloatVectorParameter
from portSim.plugins.Plugin import IPortfolioSelectionPlugin

# pylint: disable=W0201,W0613,R0201,R1710

class PortfolioGeneric(IPortfolioSelectionPlugin):

    def __init__(self):

        maxBound = 1e6

        self.settings["minCashAsset"]      = FloatParameter(0, -1, 1)
        self.settings["maxCashAsset"]      = FloatParameter(1, -1, 1)
        self.settings["minShort"]          = FloatParameter(0,  0, maxBound)
        self.settings["maxShort"]          = FloatParameter(0,  0, maxBound+1)
        self.settings["minFuturesShort"]   = FloatParameter(0,  0, 1)
        self.settings["maxFuturesShort"]   = FloatParameter(0,  0, 1)
        self.settings["longLowerBound"]    = FloatParameter(None , 0)
        self.settings["longUpperBound"]    = FloatParameter(None , 0)
        self.settings["shortLowerBound"]   = FloatParameter(None , 0)
        self.settings["shortUpperBound"]   = FloatParameter(None , 0)
        self.settings["assetUpperBound"]   = FloatVectorParameter(-1, 1)
        self.settings["assetLowerBound"]   = FloatVectorParameter(-1, 1)
        self.settings["tagUpperBound"]     = FloatVectorParameter(-1, 1)
        self.settings["tagLowerBound"]     = FloatVectorParameter(-1, 1)
        self.settings["ignoreInBounds"]    = ListParameter()
        self.settings["maxCardinality"]    = IntParameter(None, 0, maxBound)
        self.settings["minCardinality"]    = IntParameter(None, 0, maxBound)
        self.settings["minExpectedReturn"] = FloatParameter(None,-1)
        self.settings["turnoverLimit"]     = FloatParameter(None,  0.001, 2*(1 + 2*maxBound))
        self.settings["timeLimit"]         = IntParameter(60, 1, 1e6)

        self.N    = np.zeros(0)
        self.scenarios = None
        self.target    = None

        # Turnover specific
        self.turnoverLimit = None
        self.currentPositions = pd.Series(dtype = float)

        self.SOLVER_NAME = "cplex"
    # end __init__

    def execute(self, dataBundle, rebalance):
        util.stop("You must implement the function execute in a subclass of PortfolioModel or PortfolioRatioModel..")
    # end

    def collectData(self):
        util.stop("You must implement the function collectData in a subclass of PortfolioModel or PortfolioRatioModel.")
    # end collectData

    def initialise(self, admin, dataBundle):

        if len(self.rawSettings) > 0:
            util.stop("The following parameters were provided but are not valid for the %s plugin:\n\n%s"
                      % (self.__class__.__name__, "\n".join(self.rawSettings)))
        # end if

        self.settings["assetUpperBound"].process(dataBundle.data["assets"])
        self.settings["assetLowerBound"].process(dataBundle.data["assets"])
        self.settings["tagUpperBound"  ].process(dataBundle.data["tags"  ])
        self.settings["tagLowerBound"  ].process(dataBundle.data["tags"  ])
        self.settings["ignoreInBounds" ].process(dataBundle.data["assets"])

        if self.settings["minExpectedReturn"].value is not None:
            self.target = psutil.annualRateToDaily(self.settings["minExpectedReturn"].value)
        # end if

        if self.settings["minCashAsset"].value > 0 and dataBundle.settings["includeCashAsset"].value == 0:
            util.stop("Parameter minCashAsset is greater than zero (mandatory long in cash) but you set "
                      "includeCashAsset = 0.")
        # end if

        if self.settings["maxCashAsset"].value < 0 and dataBundle.settings["includeCashAsset"].value == 0:
            util.stop("Parameter maxCashAsset is less than zero (mandatory short in cash) but you set "
                      "includeCashAsset = 0.")
        # end if

        if self.settings["minCashAsset"].value > self.settings["maxCashAsset"].value:
            util.stop("Parameter maxCashAsset must be greater or equal to minCashAsset.")
        # end if

        if self.settings["minFuturesShort"].value > self.settings["maxFuturesShort"].value:
            util.stop("Parameter maxFuturesShort must be greater or equal to minFuturesShort.")
        # end if

        if self.settings["longLowerBound"].value is not None and self.settings["longUpperBound"].value is not None:
            if self.settings["longLowerBound"].value > self.settings["longUpperBound"].value:
                util.stop("Parameter longLowerBound must be greater or equal to longUpperBound.")
            # end if
        # end if

        if self.settings["shortLowerBound"].value is not None and self.settings["shortUpperBound"].value is not None:
            if self.settings["shortLowerBound"].value > self.settings["shortUpperBound"].value:
                util.stop("Parameter shortLowerBound must be greater or equal to shortUpperBound.")
            # end if
        # end if

        if self.settings["maxCardinality"].value is not None and self.settings["minCardinality"].value is not None:
            if self.settings["maxCardinality"].value < self.settings["minCardinality"].value:
                util.stop("Parameter maxCardinality must be greater or equal to minCardinality.")
            # end if
        # end if

        alb = self.settings["assetLowerBound"]
        aub = self.settings["assetUpperBound"]
        tlb = self.settings["tagLowerBound"  ]
        tub = self.settings["tagUpperBound"  ]

        for i, albValue in enumerate(alb.values):
            for j, aubValue in enumerate(aub.values):
                if alb.ids[i] == aub.ids[j] and albValue > aubValue:
                    util.stop("In the portfolio selection, assetLowerBound %.3f for %s is greater than "
                              "assetUpperBound %.3f for '%s'" % (albValue, alb.ids[i], aubValue, aub.ids[j]))
                # end if
            # end for
        # end for

        for i, tlbValue in enumerate(tlb.values):
            for j, tubValue in enumerate(tub.values):
                if tlb.ids[i] == tub.ids[j] and tlbValue > tubValue:
                    util.stop("In the portfolio selection, tagLowerBound %.3f for %s is greater than "
                              "tagUpperBound %.3f for '%s'" % (tlbValue, tlb.ids[i], tubValue, tub.ids[j]))
                # end if
            # end for
        # end for

        for i, albValue in enumerate(alb.values):
            for j, tubValue in enumerate(tub.values):
                assets = dataBundle.getAssetsInTags(tub.ids[j])
                if alb.ids[i] in assets and albValue > tubValue:
                    util.stop("In the portfolio selection, assetLowerBound %.3f was defined %s which is part of tag"
                              " '%s', for which there is a tagUpperBound of %.3f.\n"
                              "As the value is greater no feasible portfolio can ever exist."
                              % (albValue, alb.ids[i], tub.ids[j], tubValue))
                # end if
            # end for
        # end for

        # Auxiliary variables
        self.tagsAssets = dataBundle.data['tagsAssets']
        self.needsMargin =  dataBundle.needsMargin

    # end initialise


    ####################################################
    ### PUBLIC BASIC FUNCTIONS

    def calculateTurnoverLimit(self, dataBundle, assets):

        self.turnoverLimit    = None
        self.currentPositions = pd.Series(dtype = float)

        if self.settings["turnoverLimit"].value is None: return

        # Turnover limits do not apply on first and last days of the portfolio.
        initialShares = len(dataBundle.vectorSettings["initialShares"].ids) > 0        
        if (dataBundle.isNew and dataBundle.currentRebalance == 0 and not initialShares) or dataBundle.isLast: return

        self.turnoverLimit = self.settings["turnoverLimit"].value

        # All positions for all assets
        pos = dataBundle.getPositionsAsDictionary()

        # Current positions for the current assets only
        for asset in assets:
            self.currentPositions[asset] = pos[asset]
        # end

        # Mandatory turnover is the sum of weights in assets left out.
        # We have to close these positions and it doesn't matter if the
        # original weight is negative or positive, that is why we take the
        # absolute value.
        mandatoryTurnover = 0
        for asset in pos:
            if asset not in assets:
                mandatoryTurnover += abs(pos[asset])
            # end if
        # end for
        #if mandatoryTurnover > 0 and mandatoryTurnover > (self.turnoverLimit - 0.01):
        if mandatoryTurnover > 0:
            self.turnoverLimit = self.turnoverLimit + mandatoryTurnover
        # end if

    # end calculateTurnoverLimit

    def start(self, dataBundle, rebalance):
        self.scenarios = rebalance.inSampleReturns
        self.N    = rebalance.assets
        self.mu   = rebalance.expectedReturns
        if self.mu is None:
            self.mu = np.mean(self.scenarios, axis = 0).to_numpy()
        # end if
        self.mu = pd.Series(self.mu, index = self.N)
        self.status = 0
        self.weights = len(self.N)

        self.calculateTurnoverLimit(dataBundle, rebalance.assets)

    # end start

# end class PortfolioModel
