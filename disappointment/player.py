from path_search import *
from numpy import random
from copy import deepcopy
from scipy.stats import gmean
from path_cost_weights import g_weights
import gc
"""
    new strats to consider 
    - number of pieces to check which player is dominant on the board
    - another move: put something in the middle of opponent's path

"""

POSSIBLE_DIAMONDS = [[(2, -1), (0, 0), (1, -1), (1, 0)], [(0, 0), (-2, 1), (-1, 0), (-1, 1)],
                     [(1, -1), (-1, 0), (0, -1), (0, 0)], [(1, 0), (-1, 1), (0, 1), (0, 0)],
                     [(1, 0), (0, 0), (1, -1), (0, 1)], [(0, 0), (-1, 0), (0, -1), (-1, 1)],
                     [(1, -1), (0, -1), (1, -2), (0, 0)], [(0, 1), (-1, 1), (0, 0), (-1, 2)],
                     [(1, -1), (0, 0), (0, -1), (1, 0)], [(0, 0), (-1, 1), (-1, 0), (0, 1)],
                     [(0, -1), (-1, 0), (-1, -1), (0, 0)], [(1, 0), (0, 1), (0, 0), (1, 1)]]
POSSIBLE_DIAMONDS2 = [[(0, 0), (2, -1), (1, -1), (1, 0)], [(0, 0), (-2, 1), (-1, 0), (-1, 1)],
                      [(0, 0), (0, -1), (1, -1), (-1, 0)], [(0, 0), (0, 1), (1, 0), (-1, 1)],
                      [(0, 0), (1, 0), (1, -1), (0, 1)], [(0, 0), (-1, 0), (0, -1), (-1, 1)],
                      [(0, 0), (1, -2), (1, -1), (0, -1)], [(0, 0), (-1, 2), (0, 1), (-1, 1)],
                      [(0, 0), (1, -1), (0, -1), (1, 0)], [(0, 0), (-1, 1), (-1, 0), (0, 1)],
                      [(0, 0), (-1, -1), (0, -1), (-1, 0)], [(0, 0), (1, 1), (1, 0), (0, 1)]]


class Player:
    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        self.color = player
        self.n = n
        self.gameState = Graph(n, player)

        self.prevMove: Location = []
        self.currMove: Location = []

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        self.prevMove = self.currMove
        # location: Location = self.get_next_move()
        location: Location = self.alpha_beta(self.gameState)[0]
        print("action")
        print(self.prevMove)
        print(location)

        self.currMove = location
        self.game_state_eval(self.gameState, True)
        print("end action")
        self.increment_turn(self.gameState, self.gameState.player)
        return (str("PLACE"), int(location[0]), int(location[1]))

    def turn(self, player, action):
        """
        Called at the end of each player's turn to inform this player of
        their chosen action. Update your internal representation of the
        game state based on this. The parameter action is the chosen
        action itself.

        Note: At the end of your player's turn, the action parameter is
        the same as what your player returned from the action method
        above. However, the referee has validated it at this point.
        """
        cell = [0] * 2
        cell[0] = action[1]
        cell[1] = action[2]
        self.gameState.set_cell_color(tuple(cell), player)
        if player == self.color:
            self.gameState.nPlayerCells += 1
        elif player == self.gameState.enemy:
            self.gameState.nEnemyCells += 1
            enemyGameState = deepcopy(self.gameState)
            enemyGameState.flip_player_color()
            self.game_state_eval(enemyGameState, True)
            del enemyGameState
            gc.collect()
            self.increment_turn(self.gameState, player)

        self.check_diamond(cell[0], cell[1], player, self.gameState)
        return

    def increment_turn(self, gameSate: Graph, player):
        if player == red:
            gameSate.red_turn += 1
        else:
            gameSate.blue_turn += 1

    def check_diamond(self, r, c, player, gameState):
        to_clear = []
        for cells in self.generate_valid_diamonds(r, c):
            up = gameState.cell_color(cells[0])
            down = gameState.cell_color(cells[1])
            left = gameState.cell_color(cells[2])
            right = gameState.cell_color(cells[3])
            if up == down and left == right and up != left:
                if up == player or down == player:
                    to_clear.append(cells[2])
                    to_clear.append(cells[3])
                if left == player or right == player:
                    to_clear.append(cells[0])
                    to_clear.append(cells[1])
        for cell in to_clear:
            if gameState.cell_color(cell) == player:
                gameState.nPlayerCells -= 1
            elif gameState.cell_color(cell) == gameState.enemy:
                gameState.nEnemyCells -= 1
            gameState.set_cell_color(cell, empty)

    def generate_valid_diamonds(self, r, c):
        valid_diamonds = []
        for cells in POSSIBLE_DIAMONDS2:
            valid_diamond = []
            invalid = False
            for (x, y) in cells:
                if 0 <= (x + r) < self.n and 0 <= (y + c) < self.n:
                    valid_diamond.append((x + r, y + c))
                else:
                    invalid = True
                    break
            if not invalid:
                valid_diamonds.append(valid_diamond)
        return valid_diamonds

    def get_next_move(self) -> Location:

        move: Location = []
        location1: Location = []
        capture_potential = 0
        location, location_weight = self.find_path_move(self.gameState)
        location1, location1_weight = self.find_capture_move(self.gameState)
        if not location1:
            move = location
        else:
            """
                TO-DO
                at the start of game or based on number of pieces on the board set higher probability for normal a* move
                then slowly increase the probability of capture move
            """
            p_location = random.uniform(0.8, 0.2)
            choice = [0, 1]
            randomMove = random.choice(choice, 1, p=[p_location, 1 - p_location])
            print("random move: ", p_location)
            print(randomMove)
            if randomMove == choice[0]:
                move = location
            elif randomMove == choice[1]:
                move = location1
            print("location ", end=' ')
            print(location, end=' ')
            print("location1 ", end=' ')
            print(location1)
            print(move)
        return move

    def game_state_eval(self, gameState: Graph, from_main: bool):
        pathcst, _, __, cost = self.perform_path_search(gameState, True)
        if from_main:
            if gameState.player == red:
                self.gameState.red_path_costs[self.gameState.red_turn] = cost
            else:
                # print(pathcst)
                self.gameState.blue_path_costs[self.gameState.blue_turn] = cost
        else:
            if gameState.player == red:
                gameState.red_path_costs[gameState.red_turn] = cost
            else:
                gameState.blue_path_costs[gameState.blue_turn] = cost
        return cost

    def find_path_move(self, gameState: Graph):
        path_progress_contribution, path, reachedGoal, _ = self.perform_path_search(gameState, False)
        # print(path)
        move = self.get_move_from_path(reachedGoal, path, gameState)
        print("move: ", move,"ppc: ", path_progress_contribution)
        move_eval = np.NINF
        if move:
            move_eval = path_progress_contribution + self.capture_sigmoid(self.move_capture_potential(move, gameState))
        return move, move_eval

    def get_move_from_path(self, reachedGoal, path: List[Location], gameState: Graph):
        move: Location = []
        if reachedGoal:
            for cell in path:
                if gameState.cell_color(cell) == empty:
                    move = cell
                    break
        return move

    def find_capture_move(self, gameState: Graph) -> Location:
        move: Location = []
        nCaptures = 0
        for cell in gameState.playerCell:
            r = cell[0]
            c = cell[1]
            cnt = 0
            for cells in self.generate_valid_diamonds(r, c):
                up = gameState.cell_color(cells[0])
                down = gameState.cell_color(cells[1])
                left = gameState.cell_color(cells[2])
                right = gameState.cell_color(cells[3])
                if down == empty and left == right and up != left:
                    cnt = self.move_capture_potential(cells[1], gameState)
                    if cnt > nCaptures:
                        nCaptures = cnt
                        move = cells[1]
        # print('capture potential', end=' ')
        # print(nCaptures)
        path_progress_contribution = 0
        if move:
            # find path progress contribution relative to past moves
            gameStateCopy = deepcopy(gameState)
            gameStateCopy.set_cell_color(tuple(move), gameStateCopy.player)
            gameStateCopy.nPlayerCells += 1
            self.check_diamond(move[0], move[1], gameStateCopy.player, gameStateCopy)
            path_progress_contribution, _, __, ___ = self.perform_path_search(gameStateCopy, False)

        # total evaluation of this move
        move_eval = path_progress_contribution + self.capture_sigmoid(nCaptures)
        return move, move_eval

    def move_capture_potential(self, location: Location, gameState: Graph):
        r = location[0]
        c = location[1]
        cnt = 0
        for cells in self.generate_valid_diamonds(r, c):
            up = gameState.cell_color(cells[0])
            down = gameState.cell_color(cells[1])
            left = gameState.cell_color(cells[2])
            right = gameState.cell_color(cells[3])
            if up == empty and left == right and down != left and down != up:
                cnt += 1
        return cnt

    def perform_path_search(self, gameState: Graph, for_eval):
        cost = inf
        reachedGoal = False
        path: List[Location] = []
        path_progress_contribution = 0
        for start in gameState.begin:
            if gameState.cell_color(start) == gameState.enemy:
                continue
            reached_goal, came_from, cost_so_far, endGoal = a_star_search(gameState, start, gameState.goal)
            if not reached_goal:
                continue
            else:
                reachedGoal = True
                if cost_so_far[endGoal] < cost:
                    cost = cost_so_far[endGoal]
                    if not for_eval:
                        path = reconstruct_path(came_from, start, endGoal)

        if not for_eval:
            path_cost_dif = []
            turn = 0
            if gameState.player == red:
                turn = gameState.red_turn + 1
                cost_vector = cost * np.ones(gameState.red_turn + 1)
                path_cost_dif = np.subtract(gameState.red_path_costs[0:gameState.red_turn + 1], cost_vector)
                path_cost_dif = np.flip(path_cost_dif)
            elif gameState.player == blue:
                turn = gameState.blue_turn + 1
                cost_vector = cost * np.ones(gameState.blue_turn + 1)
                path_cost_dif = np.subtract(gameState.blue_path_costs[0:gameState.blue_turn + 1], cost_vector)
                path_cost_dif = np.flip(path_cost_dif)
            # path_cost_diff is used to find out how much better we are doing than previous moves
            # we now multiply them with a weight from geometric series 1/k^n and get a dot product
            relative_path_cost = np.dot(path_cost_dif, g_weights[0:turn])
            # print("relative path cost: ", relative_path_cost)
            c1 = 0.4
            c2 = 0
            path_progress_contribution = self.sigmoid(c1, c2, relative_path_cost)*relative_path_cost

        return path_progress_contribution, path, reachedGoal, cost

    def sigmoid(self, c1, c2, x):
        return 1 / (1 + np.exp(-c1 * (x - c2)))

    def capture_sigmoid(self, x):
        c1 = 0.7
        c2 = 4.6
        return self.sigmoid(c1, c2, x)

    def alpha_beta(self, state: Graph) -> Location:
        gameState = deepcopy(state)
        move0, move0_eval = self.find_path_move(gameState)
        move1, move1_eval = self.find_capture_move(gameState)
        move_idx, move_eval = self.max_value(gameState, np.NINF, np.PINF, [], 0, 0, 3)
        del gameState
        print(move_idx)
        gc.collect()
        move = move0
        if move_idx == 0:
            move = move0

        elif move_idx == 1:
            move = move1
        print(move0, move0_eval)
        print(move1, move1_eval)
        return move, move_eval

    def max_value(self, state: Graph, alpha, beta, move_index, move_eval, curr_depth, max_depth):
        print("curr_depth: max: ", curr_depth)
        if curr_depth == max_depth:
            return move_index, self.mini_max_eval(state)

        # first possible successor state
        move0, move0_eval = self.find_path_move(state)
        move1, move1_eval = self.find_capture_move(state)
        moves = [[move0, move0_eval], [move1, move1_eval]]

        best_move: Location = []
        idx =0
        best_move_index = 0
        for m, m_eval in moves:
            if not m:
                continue
            print(curr_depth, "max: ", m)
            gameState0 = deepcopy(state)
            self.apply_move(gameState0, tuple(m))
            _, min_eval0 = self.min_value(gameState0, alpha, beta, tuple(m), m_eval, curr_depth, max_depth)
            del gameState0
            gc.collect()
            if alpha < min_eval0:
                alpha = min_eval0
                best_move = _
                best_move_index = idx
            idx += 1
            if alpha >= beta:
                # pruning we dont look at other nodes
                return best_move_index, beta

        return best_move_index, alpha

    def min_value(self, state: Graph, alpha, beta, move_index, move_eval, curr_depth, max_depth):
        print("curr_depth: min ", curr_depth)
        if curr_depth == max_depth:
            return move_index, self.mini_max_eval(state)

        # first possible successor state
        move0, move0_eval = self.find_path_move(state)
        move1, move1_eval = self.find_capture_move(state)
        moves = [[tuple(move0), move0_eval], [tuple(move1), move1_eval]]

        idx = 0
        best_move: Location = []
        best_move_index = 0
        for m, m_eval in moves:
            if not m:
                continue
            print( curr_depth, "min: ", m)
            gameState0 = deepcopy(state)
            self.apply_move(gameState0, m)
            _, min_eval0 = self.max_value(gameState0, alpha, beta, m, m_eval, curr_depth + 1, max_depth)
            del gameState0
            gc.collect()
            if beta > min_eval0:
                beta = min_eval0
                best_move = _
                best_move_index = idx
            idx += 1
            if beta <= alpha:
                # pruning we dont look at other nodes
                return best_move_index, alpha

        return best_move_index, beta

    def mini_max_eval(self, state):
        cost0 = self.game_state_eval(state, False)
        gameState = deepcopy(state)
        gameState.flip_player_color()
        cost1 = self.game_state_eval(gameState, False)
        del gameState
        gc.collect()
        if self.gameState.player == state.player:
            state_eval = cost1 - cost0
        else:
            state_eval = cost0 - cost1
        return state_eval

    def apply_move(self, state: Graph, move: Location):
        state.set_cell_color(move, state.player)
        self.check_diamond(move[0], move[1], state.player, state)
        self.game_state_eval(state, True)
        # now we switch which player we are for next branch
        state.flip_player_color()
        return
