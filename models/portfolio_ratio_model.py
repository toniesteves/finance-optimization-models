import logging
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from cav.utils import util, config, log
from portSim.utils import portSimUtil as psutil
from portfolio_generic import PortfolioGeneric

# pylint: disable=W0201,W0613,R0201,R1710

class PortfolioRatioModel(PortfolioGeneric):

    def __init__(self):
        super().__init__()
        # NOTE study appropriate value for U
        self.U = 300
    #end __init__

    ####################################################
    ### PUBLIC BASIC FUNCTIONS

    def createModel(self, dataBundle):

        self.N1 = [asset for asset in self.N if self.needsMargin(asset) == 0]
        self.N2 = [asset for asset in self.N if self.needsMargin(asset) == 1]

        self.model = pyo.ConcreteModel()
        self.model.wl = pyo.Var(self.N, within = pyo.NonNegativeReals, initialize = 0)
        self.model.ws = pyo.Var(self.N, within = pyo.NonNegativeReals, initialize = 0)
        self.model.zl = pyo.Var(self.N, within = pyo.Binary          , initialize = 0)
        self.model.zs = pyo.Var(self.N, within = pyo.Binary          , initialize = 0)
        self.model.cooper = pyo.Var(within = pyo.NonNegativeReals, initialize = 0)
        self.model.zhatl = pyo.Var(self.N, within = pyo.NonNegativeReals, initialize = 0)
        self.model.zhats = pyo.Var(self.N, within = pyo.NonNegativeReals, initialize = 0)

        # Constraints that are valid for all models
        self.model.sumWeights      = pyo.Constraint(rule = self.sumWeights)
        self.model.longOrShort     = pyo.Constraint(self.N, rule = self.longOrShort)
        self.model.minShort        = pyo.Constraint(rule = self.minShort)
        self.model.maxShort        = pyo.Constraint(rule = self.maxShort)
        self.model.linkLong        = pyo.Constraint(self.N, rule = self.linkLong)
        self.model.linkShort       = pyo.Constraint(self.N1, rule = self.linkShort)
        self.model.linkShortFut    = pyo.Constraint(self.N2, rule = self.linkShortFut)
        if len(self.N2) > 0:
            self.model.minFuturesShort = pyo.Constraint(rule = self.minFuturesShort)
            self.model.maxFuturesShort = pyo.Constraint(rule = self.maxFuturesShort)
        # end if

        theresCash = config.CASH_LEND_ID in self.N and config.CASH_BORROW_ID in self.N
        ignoreInBounds = self.settings['ignoreInBounds'].values

        if theresCash:
            self.model.borrowCashNegative = pyo.Constraint(rule = self.borrowCashNegative)
            self.model.lendCashPositive   = pyo.Constraint(rule = self.lendCashPositive)
            self.model.onlyLendOrBorrow   = pyo.Constraint(rule = self.onlyLendOrBorrow)
            self.model.maxCashPositive    = pyo.Constraint(rule = self.maxCashPositive)
            self.model.maxCashNegative    = pyo.Constraint(rule = self.maxCashNegative)
            self.model.minCashPositive    = pyo.Constraint(rule = self.minCashPositive)
            self.model.minCashNegative    = pyo.Constraint(rule = self.minCashNegative)
            ignoreInBounds.append(config.CASH_LEND_ID)
            ignoreInBounds.append(config.CASH_BORROW_ID)
        # end if

        if self.target is not None:
            self.model.expectedReturn = pyo.Constraint(rule = self.minExpectedReturn)
        # end if

        self.assetsWithBounds = [asset for asset in self.N if asset not in ignoreInBounds]

        if self.settings["maxShort"].value == 0:
            self.model.noShort  = pyo.Constraint(self.N1, rule = self.noShort)
        # end if

        if self.settings["maxFuturesShort"].value == 0:
            self.model.noShortFutures  = pyo.Constraint(self.N2, rule = self.noShort)
        # end if

        if self.settings["longLowerBound"].value is not None:
            self.model.longLowerBound  = pyo.Constraint(self.assetsWithBounds, rule = self.longLowerBound)
        # end if
        if self.settings["longUpperBound"].value is not None:
            self.model.longUpperBound  = pyo.Constraint(self.assetsWithBounds, rule = self.longUpperBound)
        # end if
        if self.settings["shortLowerBound"].value is not None:
            self.model.shortLowerBound = pyo.Constraint(self.assetsWithBounds, rule = self.shortLowerBound)
        # end if
        if self.settings["shortUpperBound"].value is not None:
            self.model.shortUpperBound = pyo.Constraint(self.assetsWithBounds, rule = self.shortUpperBound)
        # end if

        # Asset and tag specific bounds
        self.addAssetLowerBound()
        self.addAssetUpperBound()
        self.addTagLowerBound(dataBundle)
        self.addTagUpperBound(dataBundle)

        if self.settings["minCardinality"].value is not None:
            self.model.minCardinality = pyo.Constraint(rule = self.minCardinality)
        # end if
        if self.settings["maxCardinality"].value is not None:
            self.model.maxCardinality = pyo.Constraint(rule = self.maxCardinality)
        # end if

        if self.turnoverLimit is not None:
            self.Nrisky  = [asset for asset in self.N if asset not in ["CASHLEND", "CASHBORROW"]]
            self.model.t = pyo.Var(self.Nrisky, within = pyo.NonNegativeReals, initialize = 0)
            self.model.turnover1   = pyo.Constraint(self.Nrisky, rule = self.turnover1)
            self.model.turnover2   = pyo.Constraint(self.Nrisky, rule = self.turnover2)
            self.model.turnoverMax = pyo.Constraint(rule = self.turnoverMax)
        # end if

        self.model.zhatl1 = pyo.Constraint(self.N, rule = self.zhatl1)
        self.model.zhatl2 = pyo.Constraint(self.N, rule = self.zhatl2)
        self.model.zhatl3 = pyo.Constraint(self.N, rule = self.zhatl3)
        self.model.zhats1 = pyo.Constraint(self.N, rule = self.zhats1)
        self.model.zhats2 = pyo.Constraint(self.N, rule = self.zhats2)
        self.model.zhats3 = pyo.Constraint(self.N, rule = self.zhats3)
    # end createModel
    #############################################################

    def optimise(self):
        
        solver = pyo.SolverFactory(self.SOLVER_NAME)

        if self.settings["timeLimit"].value is not None:
            if self.SOLVER_NAME == "glpk":
                solver.options['tmlim'] = self.settings["timeLimit"].value
            elif self.SOLVER_NAME == 'cbc':
                solver.options['seconds'] = self.settings["timeLimit"].value
            elif self.SOLVER_NAME == 'cplex':
                solver.options['timelimit'] = self.settings["timeLimit"].value
            elif self.SOLVER_NAME == 'gurobi':
                solver.options['TimeLimit'] = self.settings["timeLimit"].value
            elif self.SOLVER_NAME == 'couene':
                # Precisa criar um arquivo para o Couene ler
                pass
            # end if
        # end if

        #self.model.write("modelExported.lp")
        # This silences Pyomo warnings when the model is infeasible
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        self.results = solver.solve(self.model, tee = False)

        self.solverTerminationCondition = self.results['Solver'][0]['Termination condition']

        if self.solverTerminationCondition in ("infeasible", "unbounded"):
            self.weights = np.repeat(None, len(self.N))
            self.status = 1
        else:
            self.getWeights()
        # end if
    # end optimise


    def getWeights(self):
        weights = []
        count = 0
        cooper = pyo.value(self.model.cooper)
        maxX = 0
        for i in self.N:
            count += pyo.value(self.model.zl[i])
            count += pyo.value(self.model.zs[i])
            x = pyo.value(self.model.wl[i] - self.model.ws[i])
            weights.append(x / cooper)
            maxX = max(x, maxX);
        # end for
        log.fdebug("After solving, value of COOPER = %g (max X = %g)" % (cooper, maxX), level = 4)
        self.weights = np.array(weights)
        self.status = 1 if None in self.weights else 0
    # end getWeights


    #############################################################
    # Constraints

    def sumWeights(self, model):
        return sum(model.wl[j] - model.ws[j] for j in self.N1) + sum(model.wl[j] + model.ws[j] for j in self.N2) == model.cooper
    # end sumWeights

    def minShort(self, model):
        return sum(model.ws[j] for j in self.N1) >= model.cooper * self.settings["minShort"].value
    # end minShort

    def maxShort(self, model):
        return sum(model.ws[j] for j in self.N1) <= model.cooper * self.settings["maxShort"].value
    # end maxShort

    def longOrShort(self, model, j):
        return model.zl[j] + model.zs[j] <= 1
    # end longOrShort

    def linkLong(self, model, j):
        return model.wl[j] <= model.zhatl[j] * (1 + self.settings["maxShort"].value)
    # end linkLong

    def linkShort(self, model, j):
        return model.ws[j] <= model.zhats[j] * self.settings["maxShort"].value
    # end linkShort

    def linkShortFut(self, model, j):
        return model.ws[j] <= model.zhats[j] * (1 + self.settings["maxShort"].value)
    # end linkShortFut

    def minFuturesShort(self, model):
        return ((model.cooper + sum(model.ws[j] for j in self.N1)) * self.settings["minFuturesShort"].value
                <= sum(model.ws[j] for j in self.N2))
    # end minFuturesShort

    def maxFuturesShort(self, model):
        return ((model.cooper + sum(model.ws[j] for j in self.N1)) * self.settings["maxFuturesShort"].value
                >= sum(model.ws[j] for j in self.N2))
    # end maxFuturesShort

    def borrowCashNegative(self, model):
        return model.wl["CASHBORROW"] == 0
    # end borrowCashNegative

    def lendCashPositive(self, model):
        return model.ws["CASHLEND"] == 0
    # end lendCashPositive

    def onlyLendOrBorrow(self, model):
        return model.zl["CASHLEND"] + model.zs["CASHBORROW"] <= 1
    # end maxCashNegative

    def maxCashPositive(self, model):
        return (model.wl["CASHLEND"] <= (model.cooper + sum(model.ws[j] for j in self.N1))
                * max(0, self.settings["maxCashAsset"].value))
    # end maxCashPositive

    def maxCashNegative(self, model):
        return (model.ws["CASHBORROW"] >= sum(model.ws[j] for j in self.N1)
                * max(0, -self.settings["maxCashAsset"].value))
    # end maxCashNegative

    def minCashPositive(self, model):
        return (model.wl["CASHLEND"] >= (model.cooper + sum(model.ws[j] for j in self.N1))
                * max(0, self.settings["minCashAsset"].value))
    # end minCashPositive

    def minCashNegative(self, model):
        return (model.ws["CASHBORROW"] <= sum(model.ws[j] for j in self.N1)
                * max(0, -self.settings["minCashAsset"].value))
    # end minCashNegative

    # NOTE: Here cash assets are included in the cardinality

    def minCardinality(self, model):
        return sum(model.zl[j] + model.zs[j] for j in self.N) >= self.settings["minCardinality"].value
    # end minCardinality

    def maxCardinality(self, model):
        return sum(model.zl[j] + model.zs[j] for j in self.N) <= self.settings["maxCardinality"].value
    # end maxCardinality

    def noShort(self, model, i):
        return model.zs[i] == 0
    # end noShort

    def longLowerBound(self, model, j):
        return model.wl[j] >= model.zhatl[j] * self.settings["longLowerBound"].value
    # end longLowerBound

    def longUpperBound(self, model, j):
        return model.wl[j] <= model.zhatl[j] * self.settings["longUpperBound"].value
    # end longUpperBound

    def shortLowerBound(self, model, j):
        return model.ws[j] >= model.zhats[j] * self.settings["shortLowerBound"].value
    # end shortLowerBound

    def shortUpperBound(self, model, j):
        return model.ws[j] <= model.zhats[j] * self.settings["shortUpperBound"].value
    # end shortUpperBound

    def minExpectedReturn(self, model):
        return sum(self.mu[j] * (model.wl[j] - model.ws[j]) for j in self.N) >= model.cooper * self.target
    # end minExpectedReturn

    def turnover1(self, model, i):
        return model.t[i] >= model.cooper * self.currentPositions[i]  - (model.wl[i] - model.ws[i])
    # end turnover1

    def turnover2(self, model, i):
        return model.t[i] >= (model.wl[i] - model.ws[i]) - model.cooper * self.currentPositions[i]
    # end turnover2

    def turnoverMax(self, model):
        return sum( model.t[i] for i in self.Nrisky) <= model.cooper * self.turnoverLimit
    # end turnoverMax


    # NOTE If a certain asset is not in this rebalance, the limits are ignored,
    #      both lower and upper bounds.

    def addAssetLowerBound(self):
        assets = []
        bounds = []

        for i, asset in enumerate(self.settings["assetLowerBound"].ids):
            if asset in self.N:
                assets.append(asset)
                bounds.append(self.settings["assetLowerBound"].values[i])
            # end if
        # end for
        if len(bounds) == 0: return

        def assetLowerBound(model, i):
            if bounds[i] >= 0:
                return model.wl[assets[i]] >= bounds[i] * (model.cooper + sum(model.ws[j] for j in self.N1))
            # end if
            if assets[i] in self.N1:
                return model.ws[assets[i]] <= -bounds[i] * (sum(model.ws[j] for j in self.N1))
            # end if
            return model.ws[assets[i]] <= -bounds[i] * (sum(model.ws[j] for j in self.N2))
        # end assetUpperBound

        self.model.assetLowerBound = pyo.Constraint(range(len(assets)), rule = assetLowerBound)
    # end addAssetLowerBound

    def addAssetUpperBound(self):
        assets = []
        bounds = []
        
        for i, asset in enumerate(self.settings["assetUpperBound"].ids):
            if asset in self.N:
                assets.append(asset)
                bounds.append(self.settings["assetUpperBound"].values[i])
            # end if
        # end for
        if len(bounds) == 0: return

        def assetUpperBound(model, i):
            if bounds[i] >= 0:
                return model.wl[assets[i]] <= bounds[i] * (model.cooper + sum(model.ws[j] for j in self.N1))
            # end if
            if assets[i] in self.N1:
                return model.ws[assets[i]] >= -bounds[i] * (sum(model.ws[j] for j in self.N1))
            # end if
            return model.ws[assets[i]] >= -bounds[i] * (sum(model.ws[j] for j in self.N2))
        # end assetUpperBound

        self.model.assetUpperBound = pyo.Constraint(range(len(assets)), rule = assetUpperBound)
    # end addAssetUpperBound


    # NOTE If a certain tag has no assets in this rebalance, the limits are ignored,
    #      both lower and upper bounds.

    def addTagLowerBound(self, dataBundle):
        tagAssets = []
        bounds    = []
        tags      = []
        for i, tag in enumerate(self.settings["tagLowerBound"].ids):
            assets = dataBundle.data["tagsAssets"][tag]
            assets = [asset for asset in assets if asset in self.N]
            if len(assets) > 0:
                tagAssets.append(assets)
                bounds.append(self.settings["tagLowerBound"].values[i])
                tags.append(tag)
            # end if
        # end for
        if len(bounds) == 0: return

        def tagLowerBound(model, i):
            if bounds[i] >= 0:
                return (sum(model.wl[asset] for asset in tagAssets[i]) >= bounds[i]
                        * (model.cooper + sum(model.ws[j] for j in self.N1)))
            # end if
            allInN1 = all(elem in self.N1 for elem in tagAssets[i])
            allInN2 = all(elem in self.N2 for elem in tagAssets[i])
            if allInN1:
                return (sum(model.ws[asset] for asset in tagAssets[i]) <= -bounds[i]
                        * (sum(model.ws[j] for j in self.N1)))
            #end if
            if allInN2:
                return (sum(model.ws[asset] for asset in tagAssets[i]) <= -bounds[i]
                        * (sum(model.ws[j] for j in self.N2)))
            # end if
            util.stop("Sorry, but you defined a negative tagLowerBound for tag %s.\n"
                      "However, this tag contains both margin and non-margin assets.\n"
                      "Negative lower limits in this case are not allowed at the moment."
                      % tags[i])
        # end tagLowerBound

        self.model.tagLowerBound = pyo.Constraint(range(len(tagAssets)), rule = tagLowerBound)
    # end addTagLowerBound


    def addTagUpperBound(self, dataBundle):
        tagAssets = []
        bounds    = []
        tags      = []
        for i, tag in enumerate(self.settings["tagUpperBound"].ids):
            assets = dataBundle.data["tagsAssets"][tag]
            assets = [asset for asset in assets if asset in self.N]
            if len(assets) > 0:
                tagAssets.append(assets)
                bounds.append(self.settings["tagUpperBound"].values[i])
                tags.append(tag)
            # end if
        # end for
        if len(bounds) == 0: return

        def tagUpperBound(model, i):
            if bounds[i] >= 0:
                return (sum(model.wl[asset] for asset in tagAssets[i]) <= bounds[i]
                        * (model.cooper + sum(model.ws[j] for j in self.N1)))
            # end if
            allInN1 = all(elem in self.N1 for elem in tagAssets[i])
            allInN2 = all(elem in self.N2 for elem in tagAssets[i])
            if allInN1:
                return (sum(model.ws[asset] for asset in tagAssets[i]) >= -bounds[i]
                        * (sum(model.ws[j] for j in self.N1)))
            #end if
            if allInN2:
                return (sum(model.ws[asset] for asset in tagAssets[i]) >= -bounds[i]
                        * (sum(model.ws[j] for j in self.N2)))
            # end if
            util.stop("Sorry, but you defined a negative tagUpperBound for tag %s.\n"
                      "However, this tag contains both margin and non-margin assets.\n"
                      "Negative upper limits in this case are not allowed at the moment."
                      % tags[i])
        # end tagUpperBound

        self.model.tagUpperBound = pyo.Constraint(range(len(tagAssets)), rule = tagUpperBound)
    # end addTagUpperBound

    def zhatl1(self, model, i):
        return model.zhatl[i] <= self.U * model.zl[i]
    # end zhatl1

    def zhatl2(self, model, i):
        return model.zhatl[i] <= model.cooper
    # end zhatl2

    def zhatl3(self, model, i):
        return model.cooper - model.zhatl[i] + self.U * model.zl[i] <= self.U
    # end zhatl3

    def zhats1(self, model, i):
        return model.zhats[i] <= self.U * model.zs[i]
    # end zhats1

    def zhats2(self, model, i):
        return model.zhats[i] <= model.cooper
    # end zhats2

    def zhats3(self, model, i):
        return model.cooper - model.zhats[i] + self.U * model.zs[i] <= self.U
    # end zhats3

# end class PortfolioRatioModel
