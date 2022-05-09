import gc
from collections import OrderedDict as odict
from copy import deepcopy
from itertools import islice
import time
from disappointment.path_cost_weights import g_weights
from disappointment.path_search import *
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
        self.timespent = 0
        self.times = []
        self.prevMove: Location = []
        self.currMove: Location = []
        self.maxMoveTime = n + n//2 # 15
        self.cutOffScore = 1.48
        self.total_time_spent = 0
        self.maxtime = n * n
        self.maxPlaces = n * n
        self.time_threshold = self.maxtime - self.maxMoveTime
        self.max_depth = 1
        self.red_first_move = []
        self.first_blue_move = []
        self.increment_minimax_depth_p = 17

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        start = time.process_time()
        self.prevMove = self.currMove
        first_move = False
        if self.gameState.player == blue:
            if self.gameState.blue_turn == 0:
                first_move = True
                first_location: Location = self.red_first_move
        elif self.gameState.player == red:
            if self.gameState.red_turn == 0:
                first_move = True
                first_location: Location = (0, self.n - 1)
                self.red_first_move = first_location
        if first_move:
            location = first_location
        elif self.gameState.player == blue and self.gameState.blue_turn == 1:
            self.find_first_two_blue_move()
            location = self.first_blue_move
        else:
            # t0 = time.clock()
            location, evaluation = self.get_next_move()
            # print("evaluation: ", evaluation, "move: ", location)
            if (evaluation <= self.cutOffScore or not location) and self.total_time_spent < self.time_threshold:
                location: Location = self.alpha_beta(self.gameState)[0]
                print(location)
            # t1 = time.clock()
            # t = t1 - t0
            # self.timespent += t
            # self.times.append(t)
        #     print("time taken: ", t, "average move time: ", np.average(self.times))
        #     print("total time spent: ", self.timespent)
        # print("action")
        # print(self.prevMove)
        # print(location)
        if not location:
            location = self.get_one_position_near_enemy(self.gameState)
        self.currMove = location
        self.game_state_eval(self.gameState, True)
        # print("end action")
        self.increment_turn(self.gameState, self.gameState.player)
        end = time.process_time()
        self.total_time_spent += end - start
        if first_move and self.gameState.player == blue:
            return (str("STEAL"), )
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
        if action[0] == 'STEAL':
            cell[0] = self.red_first_move[1]
            cell[1] = self.red_first_move[0]
            self.gameState.set_cell_color(tuple(self.red_first_move), empty)
            self.gameState.set_cell_color(tuple(cell), player)
        else:
            cell[0] = action[1]
            cell[1] = action[2]
            self.gameState.set_cell_color(tuple(cell), player)
        if player == self.color:
            self.gameState.nPlayerCells += 1
            self.game_state_eval(self.gameState, True)
        elif player == self.gameState.enemy:
            enemyGameState = deepcopy(self.gameState)
            enemyGameState.flip_player_color()
            cost = self.game_state_eval(enemyGameState, True)
            if action[0] == 'STEAL':
                self.gameState.nPlayerCells -= 1
                self.gameState.red_path_costs[self.gameState.red_turn-1] = self.n
            self.gameState.nEnemyCells += 1
            del enemyGameState
            gc.collect()
            if player == red:
                if self.gameState.player == blue and self.gameState.red_turn == 0:
                    self.red_first_move = tuple(cell)
                self.gameState.red_path_costs[self.gameState.red_turn] = cost
            else:
                self.gameState.blue_path_costs[self.gameState.blue_turn] = cost
            self.increment_turn(self.gameState, player)

        self.check_diamond(cell[0], cell[1], player, self.gameState)
        # print_board(self.n, self.gameState.cell, "internal board", ansi=False)
        return

    def increment_turn(self, gameSate: Graph, player):
        # total number of pieces on the board
        n_pieces = len(gameSate.red_cells) + len(gameSate.blue_cells)
        if player == red:
            gameSate.red_turn += 1
            if self.gameState.player == red and n_pieces != 0 and n_pieces % self.increment_minimax_depth_p:
                self.max_depth += 1
        else:
            gameSate.blue_turn += 1
            if self.gameState.player == blue and n_pieces != 0 and n_pieces % self.increment_minimax_depth_p:
                self.max_depth += 1

    def find_first_two_blue_move(self):
        path_progress_contribution, path, reachedGoal, _ = self.perform_path_search(self.gameState, False, True)
        # print(path)
        path.reverse()
        move = self.get_move_from_path(reachedGoal, path, self.gameState)
        self.first_blue_move = tuple(move)

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
        move_eval = -inf
        location1: Location = []
        capture_potential = 0
        location, location_weight = self.find_path_move(self.gameState)
        location1, location1_weight = self.find_capture_move(self.gameState)

        if not location1:
            move = location
            move_eval = location_weight
        else:
            """
                TO-DO
                at the start of game or based on number of pieces on the board set higher probability for normal a* move
                then slowly increase the probability of capture move
            """
            # p_location = random.uniform(0.8, 0.2)
            # choice = [0, 1]
            # randomMove = random.choice(choice, 1, p=[p_location, 1 - p_location])
            # print("random move: ", p_location)
            # print(randomMove)
            if location_weight >= location1_weight:
                move, move_eval = location, location_weight
            else:
                move, move_eval = location1, location1_weight
        # print("location ", end=' ')
        # print(location, location_weight, end=' ')
        # print("location1 ", end=' ')
        # print(location1, location1_weight)
        # print(move, move_eval)
        return move, move_eval

    def game_state_eval(self, gameState: Graph, from_main: bool):

        if from_main:
            pathcst, _, __, cost = self.perform_path_search(gameState, True, False)
            if gameState.player == red:
                self.gameState.red_path_costs[self.gameState.red_turn] = cost
            else:
                # print(pathcst)
                self.gameState.blue_path_costs[self.gameState.blue_turn] = cost
            return cost
        else:
            pathcst, path, __, cost = self.perform_path_search(gameState, False, False)
            if gameState.player == red:
                gameState.red_path_costs[gameState.red_turn] = cost
            else:
                gameState.blue_path_costs[gameState.blue_turn] = cost
            return cost, path

    def find_path_move(self, gameState: Graph):
        path_progress_contribution, path, reachedGoal, _ = self.perform_path_search(gameState, False, True)
        # print(path)
        move = self.get_move_from_path(reachedGoal, path, gameState)
        # print(move)

        gameState0 = deepcopy(gameState)
        gameState0.set_cell_color(move, gameState0.player)
        self.check_diamond(move[0], move[1], gameState0.player, gameState0)
        __, ___, ____, applied_cost = self.perform_path_search(gameState0, False, True)
        if applied_cost == 0:
            return move, 1000
        del gameState0
        gc.collect()
        gameState0 = deepcopy(gameState)
        gameState0.flip_player_color()

        path_progress_contribution1, path, reachedGoal, _ = self.perform_path_search(gameState0, False, True)
        move1 = self.get_move_from_path(reachedGoal, path, gameState0)
        # print(path_progress_contribution1)
        gameState0.set_cell_color(move, gameState0.player)
        self.check_diamond(move1[0], move1[1], gameState0.player, gameState0)

        __, ___, ____, applied_cost = self.perform_path_search(gameState0, False, True)
        del gameState0
        gc.collect()
        if applied_cost == 0:
            return move, 1000
        # print(move)
        move_eval = path_progress_contribution
        if move == move1:
            move_eval = path_progress_contribution+path_progress_contribution1
        # print("move: ", move,"ppc: ", path_progress_contribution)
        elif not move:
            move = move1
        else:
            move_eval = path_progress_contribution1
        if move:
            # print(self.capture_sigmoid(self.move_capture_potential(move, gameState)))
            move_eval += self.capture_tanh(self.move_capture_potential(move, gameState))
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

        def is_against_wall_cell(cell):
            against_bound = False
            if cell[0] == 0 or cell[0] == self.n - 1 or cell[1] == 0 or cell[1] == self.n - 1:
                against_bound = True
            return against_bound

        def_move = []
        def_move_eval = 3
        for cell in gameState.get_player_cells():
            r = cell[0]
            c = cell[1]
            cnt = 0
            # print("capture cells; ", r, c)

            for cells in self.generate_valid_diamonds(r, c):
                up = gameState.cell_color(cells[0])
                down = gameState.cell_color(cells[1])
                left = gameState.cell_color(cells[2])
                right = gameState.cell_color(cells[3])

                if up == down and left != right and up != left and up != empty:
                    # defensive move
                    if left == empty and is_against_wall_cell(cells[2]):
                        def_move = cells[2]
                        # return cells[2], 3
                    elif right == empty and is_against_wall_cell(cells[3]):
                        def_move = cells[3]
                        # return cells[3], 3

                if down == empty and left == right and up != left and left != empty and up != down:
                    # print("cells qualified", cells[1])
                    cnt = self.move_capture_potential(cells[1], gameState)
                    # print(cnt)
                    if cnt > nCaptures:
                        nCaptures = cnt
                        move = cells[1]
        # print('capture potential', nCaptures, "move: ", move)
        path_progress_contribution = 0
        if move:
            # find path progress contribution relative to past moves
            gameStateCopy = deepcopy(gameState)
            gameStateCopy.set_cell_color(tuple(move), gameStateCopy.player)
            gameStateCopy.nPlayerCells += 1
            self.check_diamond(move[0], move[1], gameStateCopy.player, gameStateCopy)
            path_progress_contribution, _, __, ___ = self.perform_path_search(gameStateCopy, False, True,
                                                                              from_find_capture_move=True)
            # print("move colour: ", self.gameState.cell_color(move))
        # total evaluation of this move
        move_eval = path_progress_contribution + self.capture_tanh(nCaptures)
        if move_eval < self.cutOffScore and def_move:
            move_eval = def_move_eval
            move = def_move
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

    def perform_path_search(self, gameState: Graph, for_eval, for_path_move, from_find_capture_move=False):
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
                newcost = max(cost_so_far[endGoal], len(path))
                # if self.gameState.player == blue and for_path_move:
                #     # print(len(path), path)
                if newcost < cost or len(path) == 1:
                    cost = newcost
                    if not for_eval or for_path_move:
                        path = gameState.reconstruct_path(came_from, start, endGoal)
                        # print("cost_start: ", cost_so_far[start], "cost_goal: ", cost, "len path", len(path))
                        cost = max(cost, len(path))


        if for_path_move:
            # print("final: ", cost, path)
            path_cost_dif = []
            turn = 0
            if gameState.player == red:
                turn = gameState.red_turn
                # print(turn)
                cost_vector = (cost-1) * np.ones(turn)
                path_cost_dif = np.subtract(gameState.red_path_costs[0:turn], cost_vector)

                # print(path_cost_dif)
                # print(gameState.red_path_costs[0:turn])
                # print(cost_vector)
                path_cost_dif = np.flip(path_cost_dif, 0)
            elif gameState.player == blue:
                turn = gameState.blue_turn
                # print(turn)
                cost_vector = (cost-1) * np.ones(turn)
                path_cost_dif = np.subtract(gameState.blue_path_costs[0:turn], cost_vector)
                # print(path_cost_dif)
                # print(gameState.blue_path_costs[0:turn])
                # print(cost_vector)
                path_cost_dif = np.flip(path_cost_dif, 0)
                # print(path_cost_dif)
            # path_cost_dif tells me how much closer to the goal I am compared to previous moves
            # path_cost_diff is used to find out how much better we are doing than previous moves
            # we now multiply them with a weight from geometric series 1/k^n and get a dot product
            relative_path_cost = np.dot(path_cost_dif, g_weights[0:turn])
            # print("relative path cost: ", relative_path_cost)
            c1 = 0.4
            c2 = 0
            path_progress_contribution = self.sigmoid(relative_path_cost, c1, c2) * relative_path_cost

            # if cost == 2:
            #     for move in path:
            #         for cell in gameState.neighbors(move):
            #             if gameState.cell_color(cell) == gameState.player:
            #                 path_progress_contribution = 1000
            #                 if from_find_capture_move:
            #                     path_progress_contribution = 500
            #                 break
            #         if path_progress_contribution == 1000 or path_progress_contribution == 500:
            #             break

            if cost <= 1:
                # print(cost, path)
                path_progress_contribution = 1000
                if from_find_capture_move:
                    path_progress_contribution = 500
            # print("path_progess: ", path_progress_contribution)

        return path_progress_contribution, path, reachedGoal, cost

    def sigmoid(self, c1, c2, x):
        return 1 / (1 + np.exp(-c1 * (x - c2)))


    def capture_tanh(self, x):
        c1 = 0.4
        c2 = 0
        a = 4.1
        return a * np.tanh(c1 * (x - c2))

    def alpha_beta(self, state: Graph) -> Location:
        gameState = deepcopy(state)
        start_timer = time.process_time()
        move_idx, best_move, move_eval = self.max_value(gameState, np.NINF, np.PINF, 0, [], 0, self.max_depth,
                                                        start_timer)
        del gameState
        # print("alpha_beta: alpha: ", move_eval, best_move)
        gc.collect()
        return best_move, move_eval

    def max_value(self, state: Graph, alpha, beta, move_index, b_move, curr_depth, max_depth, start_timer):
        # print("curr_depth: ", curr_depth, " max: ")
        # print_board(self.n, state.cell, "max_: " + str(curr_depth), ansi=False)
        # if curr_depth == max_depth:
        #     return move_index, b_move, self.mini_max_eval(state)

        # first possible successor state
        # move0, move0_eval = self.find_path_move(state)
        # move1, move1_eval = self.find_capture_move(state)
        # moves = [[move0, move0_eval], [move1, move1_eval]]
        moves = self.move_space(state)
        best_move: Location = []
        idx = 0
        best_move_index = 0

        if curr_depth == 0:
            if state.player == blue:
                turn = state.blue_turn + 1
            else:
                turn = state.red_turn
            if turn >= self.n:
                gameState0 = deepcopy(state)
                cost0, path = self.game_state_eval(gameState0, False)
                if cost0 <= 1:
                    # print(cost0, "my moves to play: ", path)
                    if path[0]:
                        return best_move_index, path[0], alpha
                gameState0.flip_player_color()
                cost1, path = self.game_state_eval(gameState0, False)
                # enemy is about to finish their path, go block it
                if cost1 <= 1:
                    # print("depth: ", curr_depth, cost1, "enemy moves to play: ", path, path[0])
                    if path[0]:
                        return best_move_index, path[0], alpha
                del gameState0
                gc.collect()

        for m in moves:
            if not m:
                continue

            gameState0 = deepcopy(state)
            self.apply_move(gameState0, tuple(m))
            if curr_depth == max_depth:
                min_eval0 = self.terminal_state(gameState0)
            else:
                if time.process_time() - start_timer > self.maxMoveTime:
                    # trim we reaced the peak
                    return best_move_index, best_move, alpha
                _, min_eval0 = self.min_value(gameState0, alpha, beta, tuple(m), curr_depth, max_depth, start_timer)
            del gameState0
            gc.collect()
            if alpha < min_eval0:
                alpha = min_eval0
                best_move_index = idx
                if curr_depth == 0:
                    best_move = next(islice(moves.items(), best_move_index, best_move_index + 1, 1))[0]
                    # print("best move: ", best_move)
            idx += 1
            if alpha >= beta:
                # pruning we dont look at other nodes
                return best_move_index, best_move, beta

        return best_move_index, best_move, alpha

    def min_value(self, state: Graph, alpha, beta, move_index, curr_depth, max_depth, start_timer):
        # print_board(self.n, state.cell, "min_: " + str(curr_depth), ansi=False)
        # print("curr_depth: min ", curr_depth)

        # if state.player == blue:
        #     turn = state.blue_turn + 1
        # else:
        #     turn = state.red_turn
        # if turn >= self.n:
        #     gameState0 = deepcopy(state)
        #     cost0, path = self.game_state_eval(gameState0, False)
        #     if cost0 <= 1:
        #         #print(cost0, "moves to play: ", path)
        #         return 0, alpha + 0.5
        #     gameState0.flip_player_color()
        #     cost1, path = self.game_state_eval(gameState0, False)
        #     # enemy is about to finish their path, go block it
        #     if cost1 <= 1:
        #         #print(cost0, "moves to play: ", path)
        #         return 0, alpha + 0.5

        if curr_depth == max_depth:
            return move_index, self.mini_max_eval(state)[0]

        # first possible successor state
        moves = self.move_space(state)
        # print(moves)
        # move0, move0_eval = self.find_path_move(state)
        # move1, move1_eval = self.find_capture_move(state)
        # moves = [[tuple(move0), move0_eval], [tuple(move1), move1_eval]]

        idx = 0
        best_move_index = 0
        for m in moves:
            if not m:
                continue
            # print( curr_depth, "min: ", m)
            if time.process_time() - start_timer > self.maxMoveTime:
                # trim we reaced the peak
                return best_move_index, beta
            gameState0 = deepcopy(state)
            self.apply_move(gameState0, m)
            _, __, min_eval0 = self.max_value(gameState0, alpha, beta, m, [], curr_depth + 1, max_depth, start_timer)
            del gameState0
            gc.collect()
            if beta > min_eval0:
                beta = min_eval0
                best_move_index = idx
            idx += 1
            if beta <= alpha:
                # pruning we dont look at other nodes
                return best_move_index, alpha

        return best_move_index, beta

    def terminal_state(self, state):
        # print_board(self.n, state.cell, ansi=False)
        state.flip_player_color()
        return self.mini_max_eval(state)[0]

    def mini_max_eval(self, state):
        cost0, _ = self.game_state_eval(state, False)
        gameState = deepcopy(state)
        gameState.flip_player_color()
        cost1, _ = self.game_state_eval(gameState, False)
        mycost = cost0
        del gameState
        gc.collect()
        some_large_number = 1000
        cost1 = some_large_number if cost1 == inf else cost1
        cost0 = some_large_number if cost0 == inf else cost0

        if self.gameState.player == state.player:
            state_eval = cost1 - cost0
            if cost0 == 1 and cost1 == some_large_number:
                state_eval = some_large_number
            elif cost1 == 1:
                state_eval = -some_large_number
        else:
            mycost = cost1
            state_eval = cost0 - cost1
            if cost0 == 1:
                state_eval = -some_large_number
            elif cost1 == 1 and cost0 == inf:
                # print(cost1, cost0)
                state_eval = some_large_number
            # elif cost1 == 1:
            #     state_eval = 1000
        #print("state_eval: ", state_eval)
        return state_eval, mycost

    def apply_move(self, state: Graph, move: Location):
        state.set_cell_color(move, state.player)
        self.check_diamond(move[0], move[1], state.player, state)
        # now we switch which player we are for next branch
        state.flip_player_color()
        return

    def move_space(self, state: Graph):
        moves = odict()
        self.get_neighbour_cells(state, state.get_player_cells(), moves)
        self.get_neighbour_cells(state, state.get_enemy_cells(), moves)
        # if state.player == self.gameState.player:
        #     self.add_bounds(state, state.get_player_bounds(), moves)
        # print(moves)
        # self.add_bounds(state, state.get_enemy_bounds(), moves)
        # self.add_bounds(state, state.get_player_bounds(), moves)

        return moves

    def add_bounds(self, state: Graph, bounds, moves):
        for bound in bounds:
            if set(bound).isdisjoint(state.get_player_cells()):
                continue
            for cell in bound:
                if not cell or state.cell_color(cell) != empty:
                    continue
                moves.setdefault(cell, empty)

    def get_neighbour_cells(self, state: Graph, cell_list, moves):
        for cell in cell_list:
            n_list = state.neighbors(cell)
            for neighbours1 in n_list:
                if not neighbours1 or state.cell_color(neighbours1) != empty:
                    continue
                # now we know this neighbour is a valid move
                # lets find its neighbours and add those to the list
                # we only go two level deep for enemy moves
                # if state.player == self.gameState.enemy:
                #     n2_list = state.neighbors(neighbours1)
                #     for neighbours2 in n2_list:
                #         if not neighbours2 or state.cell_color(neighbours2) != empty:
                #             continue
                #         # we add 2nd degree neighbours first and then 1st degree neighbours
                #         moves.setdefault(neighbours2, empty)
                moves.setdefault(neighbours1, empty)

    def get_one_position_near_enemy(self, state: Graph):
        cell_list = state.get_enemy_cells()
        pos = []
        for cell in cell_list:
            n_list = state.neighbors(cell)
            for neighbours1 in n_list:
                if not neighbours1 or state.cell_color(neighbours1) != empty:
                    continue
                pos = neighbours1
                break
        cell_list = state.get_player_cells()
        if not pos:
            for cell in cell_list:
                n_list = state.neighbors(cell)
                for neighbours1 in n_list:
                    if not neighbours1 or state.cell_color(neighbours1) != empty:
                        continue
                    pos = neighbours1
                    break
