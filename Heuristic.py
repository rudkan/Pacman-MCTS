# myTeam.py
# ---------
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

import random, time, util
from captureAgents import CaptureAgent
from game import Directions



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='PacmanAgent', second='GhostAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class PacmanAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Height of the map
        self.map_height = gameState.data.layout.height
        # Width of the map
        self.map_width = gameState.data.layout.width
        self.map_divider = self.my_team_centerline(gameState)
    
    # Choose Pacman Action
        
    def chooseAction(self, gameState):
        agent_state = gameState.getAgentState(self.index)
        Pacman_agent = agent_state.isPacman
        actions = gameState.getLegalActions(self.index)
        food_eaten = agent_state.numCarrying

        if Pacman_agent:
            ghost_near_agent_position = []
            for ghost in self.check_enemy_ghost_threat(gameState):
                ghost_near_agent_position.append(gameState.getAgentPosition(ghost))

            available_food = self.getFood(gameState).asList()

            for opponent in self.getOpponents(gameState):
                ghost_state = gameState.getAgentState(opponent)
                scared_time = ghost_state.scaredTimer
                if scared_time > 10:
                    action_values = []
                    for action in actions:
                        action_values.append(self.feature_calculation(gameState, action))
                    heighest_action_value = max(action_values)
                    best_available_actions = []
                    for action, value in zip(actions, action_values):
                        if value == heighest_action_value:
                            best_available_actions.append(action)
                    selected_action = random.choice(best_available_actions)
                    return selected_action

            available_food_length = len(available_food)

            if not ghost_near_agent_position and food_eaten <= 4:
                action_values = []
                for action in actions:
                    action_values.append(self.feature_calculation(gameState, action))
                heighest_action_value = max(action_values)

                best_available_actions = []
                for action, value in zip(actions, action_values):
                    if value == heighest_action_value:
                        best_available_actions.append(action)
                selected_action = random.choice(best_available_actions)

            elif available_food_length < 2 or food_eaten > 7:
                selected_action = self.heuristic_best_action(gameState)
            else:
                selected_action = self.heuristic_best_action(gameState)

        else:
            action_values = []
            for action in actions:
                value = self.defensive_action_evaluate(gameState, action)
                action_values.append(value)
            heighest_action_value = max(action_values)
            best_available_actions = []
            for action, value in zip(actions, action_values):
                if value == heighest_action_value:
                    best_available_actions.append(action)
            selected_action = random.choice(best_available_actions)

        return selected_action
    
    # Select best action
    def heuristic_best_action(self, gameState):
        actions = gameState.getLegalActions(self.index)
        best_action = None
        best_utility = float('-inf')
        my_pos = gameState.getAgentState(self.index).getPosition()

        # Check if the Pacman agent is on its half of the map
        if self.is_on_own_half(my_pos):
            # If on own half, choose actions based on heuristic evaluation
            for action in actions:
                utility = self.heuristic_evaluation(gameState, action)
                if utility > best_utility:
                    best_utility = utility
                    best_action = action
        else:
            # If not on own half, move towards own half
            best_action = self.move_to_own_half(gameState)

        return best_action
    
    # Check if the given position is on the Pacman agent's own half of the map
    def is_on_own_half(self, pos): 
        own_half_positions = [(x + 1 - 2 * self.red, y) for (x, y) in self.map_divider]
        return pos in own_half_positions

    # Move towards own half of the map
    def move_to_own_half(self, gameState):
        
        my_pos = gameState.getAgentState(self.index).getPosition()
        own_half_positions = [(x + 1 - 2 * self.red, y) for (x, y) in self.map_divider]
        min_dist = float('inf')
        best_action = None

        for action in gameState.getLegalActions(self.index):
            successor = self.get_next_state(gameState, action)
            successor_pos = successor.getAgentState(self.index).getPosition()
            if successor_pos in own_half_positions:
                # If the successor position is on own half, select this action
                return action
            else:
                # Otherwise, choose the action that minimizes distance to own half
                dist = min([self.getMazeDistance(successor_pos, pos) for pos in own_half_positions])
                if dist < min_dist:
                    min_dist = dist
                    best_action = action

        return best_action
    
    # Calculate reward by multiplying feature and weights
    def heuristic_evaluation(self, gameState, action):
        successor_state = gameState.generateSuccessor(self.index, action)
        features = self.heuristic_features(successor_state)
        weights = {'getFood': 100, 'minDistToFood': -1, 'minDistToHalf': -10, 'maxDistFromGhost': -20}
        utility = features * weights
        return utility
    
    # Calculate features
    def heuristic_features(self, gameState):
        my_pos = gameState.getAgentState(self.index).getPosition()
        food_positions = self.getFood(gameState).asList()
        half_side_positions = half_side_positions = [(x + 1 - 2 * self.red, y) for (x, y) in self.map_divider]  # Assuming you have stored the half side positions earlier
        print(f"half_side_positions:{half_side_positions}")

        features = util.Counter()
        if len(food_positions) > 0:
            features['minDistToFood'] = min([self.getMazeDistance(my_pos, food) for food in food_positions])
            if gameState.getAgentState(self.index).numCarrying > 0:
                features['getFood'] = 1

        # Calculate distance to the nearest point on Pacman's half side
        features['minDistToHalf'] = min([self.getMazeDistance(my_pos, half_pos) for half_pos in half_side_positions])

        # Calculate distance to the nearest opponent's ghost
        opponent_ghosts = self.check_enemy_ghost(gameState)
        if opponent_ghosts:
            min_dist_to_ghost = min([self.getMazeDistance(my_pos, gameState.getAgentPosition(ghost)) for ghost in opponent_ghosts])
            features['maxDistFromGhost'] = -min_dist_to_ghost  # Negative value to maximize distance
        else:
            # Sets a large default value if no ghosts are nearby, 
            features['maxDistFromGhost'] = -9999

        return features
    
    # Get next state
    def get_next_state(self, gameState, action):
        return gameState.generateSuccessor(self.index, action)

    # Check for enemy ghost nearby
    def check_enemy_ghost_threat(self, gameState):
        ghosts = self.check_enemy_ghost(gameState)
        my_agent_position = gameState.getAgentPosition(self.index)
        EnemyGhosts_Threat = []

        for ghost in ghosts:
            ghost_distance = self.getMazeDistance(my_agent_position, gameState.getAgentPosition(ghost))
            if ghost_distance <= 3:
                EnemyGhosts_Threat.append(ghost)

        return EnemyGhosts_Threat

    # Calculate minimum distance from agent position to the food
    def check_min_dist_to_food(self, gameState):
        my_agent_position = gameState.getAgentPosition(self.index)
        food_positions = self.getFood(gameState).asList()

        food_distances = []
        for food in food_positions:
            distance_to_food = self.getMazeDistance(my_agent_position, food)
            food_distances.append(distance_to_food)

        shortest_distance_to_food = min(food_distances)

        return shortest_distance_to_food

    # Calculate the centerline of the map
    def my_team_centerline(self, gameState):
        redTeam = self.red
        center_walls = gameState.getWalls().asList()
        if redTeam:
            center_x = ((self.map_width // 2) - 1)
        else:
            center_x = (self.map_width // 2)

        center_line = []

        for height in range(self.map_height):
            point_on_center = (center_x, height)

            center_line.append(point_on_center)

        center_positions = []
        for x, y in center_line:

            if (x, y) not in center_walls and (x + 1 - 2 * self.red, y) not in center_walls:
                center_positions.append((x, y))

        return center_positions

    # Calculate defensive features
    def defensive_features(self, gameState, action):
        defensive_features = util.Counter()
        successor = self.get_next_state(gameState, action)
        available_food = self.getFood(successor).asList()
        defensive_features['successorScore'] = -len(available_food)

        if len(available_food) > 0:
            current_pos = successor.getAgentState(self.index).getPosition()
            min_distance = min([self.getMazeDistance(current_pos, food) for food in available_food])
            defensive_features['distanceToFood'] = min_distance
        return defensive_features

    # Calculate features
    def feature_calculation(self, gameState, action):
        next_game_state = self.get_next_state(gameState, action)
        next_state_food = next_game_state.getAgentState(self.index).numCarrying
        food_eaten = gameState.getAgentState(self.index).numCarrying
        features_counter = util.Counter()

        if next_state_food > food_eaten:
            features_counter['getFood'] = 1
        else:
            if len(self.getFood(next_game_state).asList()) > 0:
                features_counter['minDistToFood'] = self.check_min_dist_to_food(next_game_state)
        features = features_counter

        weights = {'minDistToFood': -1, 'getFood': 100}
        return features * weights

    # Calculate weights
    def defensive_weights(self, gameState, action): 
        defensive_weights = {'successorScore': 100, 'distanceToFood': -1}
        return defensive_weights

    # Calculate the reawrd by multiplying features and weights
    def defensive_action_evaluate(self, gameState, action):
        features_value = self.defensive_features(gameState, action)  # so this is getting both defensive_feature values
        weights_value = self.defensive_weights(gameState, action)
        return features_value * weights_value

    # Check for enemy ghost nearby
    def check_enemy_ghost(self, gameState):
        EnemyGhostList = []
        for enemy_ghost in self.getOpponents(gameState):
            enemyGostState = gameState.getAgentState(enemy_ghost)

            if (enemyGostState.scaredTimer == 0) and (not enemyGostState.isPacman):
                enemyGostState_position = gameState.getAgentPosition(enemy_ghost)
                if enemyGostState_position is not None:
                    EnemyGhostList.append(enemy_ghost)

        return EnemyGhostList

# Ghost Agent Class
class GhostAgent(PacmanAgent):

    # Calculate defensive features
    def defensive_features(self, gameState, action):
        next_game_state = self.get_next_state(gameState, action)
        agent_state = next_game_state.getAgentState(self.index)

        agent_position = agent_state.getPosition()
        defensive_features = util.Counter()

        if not agent_state.isPacman:
            defensive_features['defensive'] = 1
        else:
            defensive_features['defensive'] = 0

        Opponent_list = []

        for n in self.getOpponents(next_game_state):
            opponent_state = next_game_state.getAgentState(n)
            Opponent_list.append(opponent_state)

        Opponent_pacman = []
        for opponent in Opponent_list:
            if opponent.isPacman and opponent.getPosition() is not None:
                Opponent_pacman.append(opponent)

        defensive_features['total_opponent_pacman'] = len(Opponent_pacman)

        if len(Opponent_pacman) > 0:
            Opponent_pacman_disctance = []
            for op in Opponent_pacman:
                Opponent_pacman_disctance.append(self.getMazeDistance(agent_position, op.getPosition()))
            defensive_features['op_distance'] = min(Opponent_pacman_disctance)

        if Directions.STOP == action:
            defensive_features['stop'] = 1
        back_directions = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if back_directions == action:
            defensive_features['back'] = 1

        return defensive_features
    
    # Defining defensive weights
    def defensive_weights(self, gameState, action):
        return {'total_opponent_pacman': -1000, 'defensive': 100, 'op_distance': -10, 'stop': -100, 'back': -2}