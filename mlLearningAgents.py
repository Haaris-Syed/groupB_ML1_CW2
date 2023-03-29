# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        # state features for agent to be trained on
        self.state = state
        self.pacPosition = tuple(state.getPacmanPosition())
        self.ghostPositions = tuple(state.getGhostPositions())
        self.foodLocs = tuple(tuple(i) for i in state.getFood())

    # hashing tuple of features/instance variables
    def __hash__(self):
        return hash((self.pacPosition, self.ghostPositions, self.foodLocs))

    # overload __eq__ function to compare objects based in terms of values
    def __eq__(self, other):
        if isinstance(other, GameStateFeatures):
            return (self.pacPosition, self.ghostPositions, self.foodLocs) == (other.pacPosition, other.ghostPositions, other.foodLocs)
        return False

    # supplementary methods to get legal action
    def getLegalActions(self):
        legal = self.state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        return legal

    def getGameState(self):
        return self.state


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # initialize q table and frequency table see util.Counter()
        self.QTable = util.Counter()
        self.freqTable = util.Counter()

        # initialize previous state, action, reward to None
        self.prevState = None
        self.prevAction = None
        self.prevReward = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """

        # reward signal is based on the score difference between two state
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.QTable[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """

        actions = state.getLegalActions()

        # if not terminal
        if actions:

            # get all q values from all possible legal actions and returns the largest of it
            stateQValues = [self.getQValue(state, action) for action in actions]
            stateQValues.append(self.getQValue(state, None))
            return max(stateQValues)
        else:
            return self.getQValue(state, None)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """

        # calculate the possible max q value of transitioned state
        maxQValue = self.maxQValue(nextState)

        # calculate temporal difference error
        tdError = reward + self.gamma * (maxQValue - self.getQValue(state, action))

        # update q table using update equation
        self.QTable[(state, action)] = self.QTable[(state, action)] + self.alpha * tdError

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # increment frequency table see util.Counter()
        self.freqTable[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """

        return self.freqTable[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """

        # return utility as value is counts is bigger than self.maxAttempts
        # otherwise return infinity
        if counts > self.maxAttempts:
            return utility
        else:
            return float('inf')

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """

        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # get the current game state and compute the reward signal from the previous state to the current
        currState = GameStateFeatures(state)
        rewardSignal = self.computeReward(self.prevState.getGameState(), currState.getGameState()) if self.prevState else 0

        # pick action as function of the exploration and q values
        explorationValues = [[self.explorationFn(self.getQValue(currState, action), self.getCount(currState, action)), action] for action in legal]
        action = max(explorationValues, key=lambda x: x[0])[1] if explorationValues else random.choice(legal)

        # check if state is terminal
        if not legal:
            self.QTable[(self.prevState, None)] = rewardSignal
        if self.prevState is not None:

            # update frequency table
            self.updateCount(self.prevState, self.prevAction)
            # update q table
            self.learn(self.prevState, self.prevAction, self.prevReward, currState)

        self.prevState, self.prevAction, self.prevReward = currState, action, rewardSignal

        return self.prevAction

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """

        # final update
        currState = GameStateFeatures(state)
        if self.prevState is not None:
            self.updateCount(self.prevState, self.prevAction)
            self.learn(self.prevState, self.prevAction, self.prevReward, currState)

        # final reward
        rewardSignal = self.computeReward(self.prevState.getGameState(), currState.getGameState())
        self.QTable[(currState, None)] = rewardSignal

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
