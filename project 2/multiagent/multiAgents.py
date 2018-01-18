# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodWeight = 1.0
        ghostWeight = -5.0

        # GAME SCORE
        gameScore = successorGameState.getScore()

        # GHOST SCORE
        ghostPositions = currentGameState.getGhostPositions()
        ghostScore = 0
        for ghostPosition in ghostPositions:
          pacToGhostDistance = util.manhattanDistance(newPos, ghostPosition)
          ghostScore += pacToGhostDistance

        if ghostScore != 0:
          ghostScore = 1.0/ghostScore
        else:
          ghostScore = 0

        ghostScore = ghostWeight * ghostScore

        foodList = []
        foodScore = 0
        for foodPosition in newFood.asList():
          pacToFoodDistance = manhattanDistance(newPos, foodPosition)
          foodList.append(pacToFoodDistance)

        # distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(foodList):
            foodScore += foodWeight / min(foodList)

        totalScore = gameScore + ghostScore + foodScore
        return totalScore



def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimaxSearch(self, gameState, current_depth, agentID, anchor):
          if agentID == gameState.getNumAgents():
            agentID = 0
          if current_depth == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose() or gameState.getLegalActions(agentID)==0:
            return self.evaluationFunction(gameState)

          agentActions = gameState.getLegalActions(agentID)
          minArray = []
          maxArray = []
          finalPath = []
          for agentAction in agentActions:
            nextGameState = gameState.generateSuccessor(agentID, agentAction)
            if agentID == 0 or agentID == gameState.getNumAgents():
              # I am pacman, 
              search = minimaxSearch(self, nextGameState, current_depth + 1, 1, False)
              maxArray.extend([search])
              if len(maxArray) == len(agentActions):
                return max(maxArray), agentActions[maxArray.index(max(maxArray))]
            else:
              # I am an agent
              search = minimaxSearch(self, nextGameState, current_depth + 1, agentID + 1, False)
              minArray.extend([search])
              if len(minArray) == len(agentActions):
                return min(minArray)

        value = minimaxSearch(self, gameState, 0, 0, True)
        return value[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # print "ALPHA-BETA"

        alpha = float("-inf")
        beta = float("inf")
        search = float("-inf")

        def maxAB(self, gameState, current_depth, alpha, beta):
          search = float("-inf")
          agentID = current_depth % gameState.getNumAgents()
          gameActions = gameState.getLegalActions(agentID)
          for agentAction in gameActions:
            nextGameState = gameState.generateSuccessor(agentID, agentAction)
            search = max(search, alphaBetaSearch(self, nextGameState, current_depth + 1, alpha, beta))
            if search > beta:
              return search
            alpha = max(alpha, search)
          return search

        def minAB(self, gameState, current_depth, alpha, beta):
          search = float("inf")
          agentID = current_depth % gameState.getNumAgents()
          gameActions = gameState.getLegalActions(agentID)
          for agentAction in gameActions:
            nextGameState = gameState.generateSuccessor(agentID, agentAction)
            search = min(search, alphaBetaSearch(self, nextGameState, current_depth + 1, alpha, beta))
            if search < alpha:
              return search
            beta = min(beta, search)
          return search

        def alphaBetaSearch(self, gameState, current_depth, alpha, beta):
          agentID = current_depth % gameState.getNumAgents()
          if current_depth == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose() or gameState.getLegalActions(agentID)==0:
            return self.evaluationFunction(gameState)
          elif agentID == 0:
            return maxAB(self, gameState, current_depth, alpha, beta)
          else:
            return minAB(self, gameState, current_depth, alpha, beta)

        alphaBetaSearchValues = []
        agentID = 0
        agentActions = gameState.getLegalActions(agentID)
        for agentAction in agentActions:
          nextGameState = gameState.generateSuccessor(agentID, agentAction)
          alphaBetaSearchValues.extend([alphaBetaSearch(self, nextGameState, 1, alpha, beta)])
          search = max(alphaBetaSearchValues)
          if search > beta:
            correctAction = agentActions[alphaBetaSearchValues.index(max(alphaBetaSearchValues))]
            return correctAction
          alpha = max(alpha, search)
        correctAction = agentActions[alphaBetaSearchValues.index(max(alphaBetaSearchValues))]
        return correctAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def flatten(container):
          for i in container:
              if isinstance(i, (list,tuple)):
                  for j in flatten(i):
                      yield j
              else:
                  yield i

        def expectimaxSearch(self, gameState, current_depth, agentID, anchor):
          if agentID == gameState.getNumAgents():
            agentID = 0
          if current_depth == self.depth*gameState.getNumAgents() or gameState.isWin() or gameState.isLose() or gameState.getLegalActions(agentID)==0:
            return self.evaluationFunction(gameState)

          agentActions = gameState.getLegalActions(agentID)
          minArray = []
          maxArray = []
          finalPath = []
          for agentAction in agentActions:
            nextGameState = gameState.generateSuccessor(agentID, agentAction)
            if agentID == 0 or agentID == gameState.getNumAgents():
              # I am pacman, 
              search = expectimaxSearch(self, nextGameState, current_depth + 1, 1, False)
              maxArray.extend([search])
              if len(maxArray) == len(agentActions):
                return [max(maxArray), agentActions[maxArray.index(max(maxArray))]]
            else:
              # I am an agent
              search = expectimaxSearch(self, nextGameState, current_depth + 1, agentID + 1, False)
              minArray.extend([search])
              if len(minArray) == len(agentActions):
                minArray = list(flatten(minArray))
                minArrayRemoveStrings = [x for x in minArray if not isinstance(x, basestring)]
                if len(minArrayRemoveStrings):
                  minArray = minArrayRemoveStrings
                expectation = reduce(lambda x, y: x + y, minArray) / len(minArray)
                return expectation

        value = expectimaxSearch(self, gameState, 0, 0, True)
        return value[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodWeight = 1.0
    ghostWeight = -5.0
    scaredGhostWeight = 10.0

    # GAME SCORE
    gameScore = currentGameState.getScore()

    # GHOST SCORE (POSITIONS)
    ghostPositions = currentGameState.getGhostPositions()
    ghostScore = 0
    for ghostPosition in ghostPositions:
      pacToGhostDistance = util.manhattanDistance(newPos, ghostPosition)
      ghostScore += pacToGhostDistance


    if ghostScore != 0:
      ghostScore = 1.0/ghostScore
    else:
      ghostScore = 0

    ghostScore = ghostWeight * ghostScore

    # GHOST SCORE (SCARED)
    scaredGhostScore = 0
    for ghostState in newGhostStates:
      if ghostState.scaredTimer > 0:
        foodWeight = 100.0
        distance = manhattanDistance(newPos, ghostState.getPosition())
        if distance > 0:
          scaredGhostScore += scaredGhostWeight / distance
        else:  # otherwise -> run!
          scaredGhostScore -= scaredGhostWeight / distance


    foodList = []
    foodScore = 0
    for foodPosition in newFood.asList():
      pacToFoodDistance = manhattanDistance(newPos, foodPosition)
      foodList.append(pacToFoodDistance)

    if len(foodList):
        foodScore += foodWeight / min(foodList)

    totalScore = gameScore + ghostScore + foodScore
    return totalScore

# Abbreviation
better = betterEvaluationFunction

