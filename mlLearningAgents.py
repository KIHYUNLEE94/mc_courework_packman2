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
        # Store the underlying game state so that we can
        # use it as a compact, hashable representation.
        #
        # GameState already implements __eq__ and __hash__
        # based on the essential game data, so we simply
        # wrap it here.
        self.state = state

    def __hash__(self):
        """
        Make GameStateFeatures usable as a dictionary key by
        delegating to the underlying GameState hash.
        """
        return hash(self.state)

    def __eq__(self, other):
        """
        Two feature objects are equal if the underlying game
        states are equal.
        """
        if not isinstance(other, GameStateFeatures):
            return False
        return self.state == other.state


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
        # Core learning hyperparameters
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games that have finished so far
        self.episodesSoFar = 0
        # Q-value table: maps (GameStateFeatures, action) -> Q-value
        self.qValues = util.Counter()
        # Visitation counts for (state, action) pairs
        self.counts = util.Counter()
        # Track last transition for online learning
        self.lastState = None
        self.lastAction = None

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
        # Reward is defined as the change in game score between
        # two consecutive states. The internal scoring already
        # encodes step penalties, food/ghost rewards and win/lose.
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
        return self.qValues[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # If there are no legal actions (terminal state), the
        # maximal future Q-value is defined to be 0.
        legal = state.state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        if not legal:
            return 0.0

        values = [self.getQValue(state, action) for action in legal]
        return max(values)

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
        # Standard Q-learning update:
        #
        #   Q(s, a) ← Q(s, a) + α * (r + γ * max_a' Q(s', a') − Q(s, a))
        #
        currentQ = self.getQValue(state, action)
        futureQ = self.maxQValue(nextState)
        target = reward + self.getGamma() * futureQ
        updated = currentQ + self.getAlpha() * (target - currentQ)
        self.qValues[(state, action)] = updated

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
        self.counts[(state, action)] += 1

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
        return self.counts[(state, action)]

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
        # Simple optimistic exploration: for actions that have been
        # tried fewer than maxAttempts times in a given state, boost
        # their apparent utility so they are preferred during
        # exploitation. Once sufficiently tried, fall back to the
        # given utility.
        if counts < self.getMaxAttempts():
            return utility + 1.0
        return utility

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
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Perform an online Q-learning update using the transition
        # from the previous state (if any) to the current one.
        if self.lastState is not None and self.lastAction is not None:
            prevFeatures = GameStateFeatures(self.lastState)
            reward = self.computeReward(self.lastState, state)
            self.learn(prevFeatures, self.lastAction, reward, stateFeatures)

        # Epsilon-greedy exploration/exploitation:
        #  - with probability epsilon, take a random legal action;
        #  - otherwise, pick the action with the highest exploration-
        #    adjusted Q-value using explorationFn.
        if util.flipCoin(self.epsilon):
            chosenAction = random.choice(legal)
        else:
            scored_actions = []
            for action in legal:
                q = self.getQValue(stateFeatures, action)
                c = self.getCount(stateFeatures, action)
                score = self.explorationFn(q, c)
                scored_actions.append((score, action))
            maxScore = max(scored_actions, key=lambda sa: sa[0])[0]
            bestActions = [a for (s, a) in scored_actions if s == maxScore]
            chosenAction = random.choice(bestActions)

        # Update visitation counts and remember this state/action
        # for use in the next Q-learning update.
        self.updateCount(stateFeatures, chosenAction)
        self.lastState = state
        self.lastAction = chosenAction

        return chosenAction

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Perform one final Q-learning update for the transition into
        # this terminal state so that the win/lose rewards are taken
        # into account. final() is the only place those rewards are
        # accessible for the last move.
        if self.lastState is not None and self.lastAction is not None:
            terminalFeatures = GameStateFeatures(state)
            prevFeatures = GameStateFeatures(self.lastState)
            reward = self.computeReward(self.lastState, state)
            self.learn(prevFeatures, self.lastAction, reward, terminalFeatures)
            self.lastState = None
            self.lastAction = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
