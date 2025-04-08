import pyomo.environ as pyo
from cav.utils import log
from portSim.simulation.Parameter import FloatParameter
from portfolio_ratio_model import PortfolioRatioModel

# pylint: disable=R0201

class STARRRatio(PortfolioRatioModel):

    def __init__(self):
        self.settings["alpha"] = FloatParameter(0.05, 0, 1)
        self.scenariosIndex = []

        self.SOLVER_NAME = "cplex"
    # end __init__

    ####################################################
    ### PUBLIC BASIC FUNCTIONS

    def execute(self, dataBundle, rebalance):


        log.debug("\nSolving the STARR ratio maximisation model...", level = 2)
        self.start(dataBundle, rebalance)
        self.createModel(dataBundle)

        self.optimise()
        log.debug("Solver finished with condition '%s'." % self.solverTerminationCondition, level = 2)

        self.addDBData({"status" : self.solverTerminationCondition, "numAssets" : len(self.N)})
    # end

    def createModel(self, dataBundle):
        super().createModel(dataBundle)
        self.scenariosIndex = range(self.scenarios.shape[0])

        self.model.theta = pyo.Var(within = pyo.Reals, initialize = -float('inf'))
        self.model.d     = pyo.Var(self.scenariosIndex, within = pyo.NonNegativeReals, initialize = 0.0)

        self.model.objective = pyo.Objective(rule = self.objectiveFunction, sense = pyo.maximize)

        self.model.CVaRConstraint1 = pyo.Constraint(rule = self.CVaRConstraint1)
        self.model.CVaRConstraint2 = pyo.Constraint(self.scenariosIndex, rule = self.CVaRConstraint2)
    # end createModel

    #############################
    ### STARR ratio specific constraints

    def objectiveFunction(self, model):
        return sum(self.mu[j] * (model.wl[j] - model.ws[j]) for j in self.N)
    # end objectiveFunction

    def CVaRConstraint1(self, model):
        prob = 1/self.scenarios.shape[0]
        return model.theta + 1/(self.settings["alpha"].value) * sum(prob * model.d[t] for t in self.scenariosIndex) <= 1
    # end CVaRConstraint1

    def CVaRConstraint2(self, model, t):
        return model.d[t] >= -model.theta - sum(self.scenarios[j][t] * (model.wl[j] - model.ws[j]) for j in self.N)
    # end CVaRConstraint2

    #############################

    def collectData(self):
        return self.status, self.weights
    # end collectData

# end class STARRRatio
