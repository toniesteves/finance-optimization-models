import pandas as pd
import pyomo.environ as pyo
from cav.utils import util
from portSim.simulation.Parameter import FloatParameter
from portfolio_ratio_model import PortfolioRatioModel

# pylint: disable=R0201

class SharpeRatio(PortfolioRatioModel):

    def __init__(self):

        self.SOLVER_NAME = "cplex"
    # end __init__

    ####################################################
    ### PUBLIC BASIC FUNCTIONS

    def execute(self, dataBundle, rebalance):


        util.debug("\nSolving the Sharpe ratio maximisation model...", level = 2)
        self.start(dataBundle, rebalance)
        self.createModel(dataBundle, rebalance)

        self.optimise()
        util.debug("Solver finished with condition '%s'." % self.solverTerminationCondition, level = 2)

        self.addDBData({"status" : self.solverTerminationCondition, "numAssets" : len(self.N)})
    # end

    def createModel(self, dataBundle, rebalance):
        super().createModel(dataBundle)

        if rebalance.covMatrix is None:
            util.stop("You tried executing the SharpeRatio model, but the covariance matrix has "
                      "not been defined.\nEither run the system with option quadraticData = 1 or "
                      "use an InputDataPlugin that computes the matrix.")
        # end if
        if rebalance.expectedReturns is None:
            util.stop("You tried executing the SharpeRatio model, but the vector of expected retuns has "
                      "not been defined.\nEither run the system with option quadraticData = 1 or "
                      "use an InputDataPlugin that computes the matrix.")
        # end if

        self.Sigma  = pd.DataFrame(rebalance.covMatrix, columns = self.N, index = self.N)
        self.mu = pd.Series(rebalance.expectedReturns, index = self.N, dtype = float)
        
        self.model.objective = pyo.Objective(rule = self.objectiveFunction, sense = pyo.maximize)

        self.model.returnConstraint = pyo.Constraint(rule = self.returnConstraint)

    # end createModel

    #############################
    ### Sharpe ratio specific constraints

    def objectiveFunction(self, model):
        return (sum(sum((model.wl[i] * model.wl[j] + model.wl[i] * model.ws[j] + model.ws[i] * model.wl[j] 
                + model.ws[i] * model.ws[j]) * self.Sigma[j][i] for j in self.N) for i in self.N))    
    # end objectiveFunction


    def returnConstraint(self, model):
        return sum(self.mu[j] * (model.wl[j] - model.ws[j]) for j in self.N) == 1
    # end returnConstraint

    #############################

    def collectData(self):
        return self.status, self.weights
    # end collectData

# end class SharpeRatio
