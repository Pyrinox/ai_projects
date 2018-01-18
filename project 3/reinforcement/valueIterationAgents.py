# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print "HELLO", self.iterations
        k = 0
        states = self.mdp.getStates()
        # update_values = util.Counter()
        # print "STATES: ", states
        while k < self.iterations:
          batch = util.Counter()
          for state in states:
            if not self.mdp.isTerminal(state):
              qvalues = util.Counter()
              actions = self.mdp.getPossibleActions(state)
              for action in actions:
                qvalues[action] = self.computeQValueFromValues(state, action)
              batch[state] = qvalues[qvalues.argMax()]
            # self.computeActionFromValues(state)
          self.values = batch.copy()
          k += 1
        # self.values = update_values

        # k = 0
        # states = self.mdp.getStates()
        # while k < self.iterations:
        #   update_values = util.Counter()
        #   for state in states:
        #     actions = self.mdp.getPossibleActions(state)
        #     to_max = []
        #     for action in actions:
        #       transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)
        #       #each transition state format = [state2, T-probability]
              
        #       # print "ACTION", action
        #       value = 0
        #       for transitionState in transitionStates:
        #         # print "HERE: ", transitionState
        #         reward = self.mdp.getReward(state, action, transitionState[0])
        #         # print "REWARD: ", reward
        #         to_add = transitionState[1] * (reward + (self.discount * self.getValue(transitionState[0])))
        #         value += to_add
        #       to_max.append([value])
        #       # values.extend([value])
        #       # action_values.append(values)
        #       # values[value] = action
        #     update_values[state] = max(to_max)
        #   self.values = update_values

        #   print "HERE"


        # if self.iterations == 0:
        #   return 0
        # print "HEllo"
        # print type(self.values)
        # print self.values
        # print "Iterations: ", self.iterations
        # print "discount: ", self.discount
        # print "mdp states: ", self.mdp.getStates()
        # print "getPossibleActions: ", self.mdp.getPossibleActions(state)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        # next_state_values = self.values.copy()
        for transitionState in transitionStates:
          # if self.mdp.isTerminal(transitionState[0]):
          #   return None
          # next_value = self.getValue(transitionState[0])
          # if next_value is None:

          reward = self.mdp.getReward(state, action, transitionState[0])
          to_add = transitionState[1] * (reward + (self.discount * self.getValue(transitionState[0])))
          # next_state_values[transitionState[0]] = to_add
          value += to_add
        # print "value:", value
        # self.values[state] = value
        # self.values = next_state_values
        return value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # values = []
        # print "self.values", self.values
        action_values = util.Counter()
        # tmpCopy = self.values.copy()
        # print "self values: ", self.values
        # print "action values ", action_values
        # action_values = {}
        actions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state) or len(actions) == 0:
          return None

        # print "ACTIONSKSDJFLKSDJFKSFLKS:", actions
        for action in actions:
          # print "ACTION: ", action
          # print "STATE: ", state
          
          value = self.computeQValueFromValues(state, action)
          # print "value:", value
          action_values[action] = value
          # print "update: ", action_values
        # print "HELLODLKFJ:", action_values.argMax()
        # print "HELLODLKFJ:", action_values[max(action_values)]
        # return max(action_values)
        # print "self.valus: ", action_values
        # self.values = action_values.copy()
        # print "check here:", action_values
        # key = action_values.argMax()
        # largest_value = action_values[key]
        # self.values[state] = largest_value
        return action_values.argMax()

        # print "argmax: ", self.values.argMax()
        # for state in self.values:





        # actions = self.mdp.getPossibleActions(state)
        # for action in actions:
        #   transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)
          
        #   value = 0
        #   for transitionState in transitionStates:
        #     reward = self.mdp.getReward(state, action, transitionState[0])
        #     to_add = transitionState[1] * (reward + (self.discount * self.getValue(transitionState[0])))
        #     value += to_add

        #   action_values[value] = action
        # return action_values[max(action_values)]

      


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        k = 0
        states = self.mdp.getStates()
        # update_values = util.Counter()
        # print "STATES: ", states
        # print "GUAC: ", states


        while k < self.iterations:
          # batch = util.Counter()
          current_update = states[k % len(states)]
          if not self.mdp.isTerminal(current_update):
            qvalues = util.Counter()
            actions = self.mdp.getPossibleActions(current_update)
            for action in actions:
              qvalues[action] = self.computeQValueFromValues(current_update, action)
            self.values[current_update] = qvalues[qvalues.argMax()]
          k += 1
          # k = k % len(states)


          # for state in states:
          #   if not self.mdp.isTerminal(state):
          #     qvalues = util.Counter()
          #     actions = self.mdp.getPossibleActions(state)
          #     for action in actions:
          #       qvalues[action] = self.computeQValueFromValues(state, action)
          #     self.values[state] = qvalues[qvalues.argMax()]
          #   # self.computeActionFromValues(state)
          # # self.values = batch.copy()
          # k += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        predecessors = {}
        pq = util.PriorityQueue()
        states = self.mdp.getStates()

        #initialize predecessors with states
        for state in states:
          predecessors[state] = set()

        for state in states:
          actions = self.mdp.getPossibleActions(state)
          for action in actions:
            transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)
            for transitionState in transitionStates:
              if transitionState[1] > 0:
                predecessors[transitionState[0]].add(state)

        
        track_new_values = util.Counter()
        for state in states:
          if not self.mdp.isTerminal(state):
            actions = self.mdp.getPossibleActions(state)
            # qvalue = 0
            maxQValue = []
            for action in actions:
              qvalue = self.computeQValueFromValues(state, action)
              maxQValue.extend([qvalue])
            if len(maxQValue) != 0:
              diff = abs(self.values[state] - max(maxQValue))
              # print "helluh", diff
              pq.push(state, -diff)

        k = 0
        while k < self.iterations:
          if pq.isEmpty():
            return

          state = pq.pop()
          if not self.mdp.isTerminal(state):
            maxQValue = []
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              qvalue = self.computeQValueFromValues(state, action)
              maxQValue.extend([qvalue])
            self.values[state] = max(maxQValue)


          for predecessor in predecessors[state]:
            maxQValue = []
            actions = self.mdp.getPossibleActions(predecessor)
            for action in actions:
              qvalue = self.computeQValueFromValues(predecessor, action)
              maxQValue.extend([qvalue])

            if len(maxQValue) != 0:
              diff = abs(self.values[predecessor] - max(maxQValue))
              if diff > self.theta:
                pq.update(predecessor, -diff)
          k += 1
                






