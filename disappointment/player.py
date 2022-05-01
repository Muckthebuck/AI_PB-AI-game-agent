from path_search import *
from numpy import random
from copy import  deepcopy
from scipy.stats import gmean
from path_cost_weights import  g_weights
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
        location: Location = self.get_next_move()
        print(self.prevMove)
        print(location)
        self.currMove = location
        self.gameState.turn += 1
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

        self.check_diamond(cell[0], cell[1], player, self.gameState)
        return

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

    def find_path_move(self, gameState: Graph):
        path_progress_contribution, path, reachedGoal = self.perform_path_search(gameState)
        print(path)
        move = self.get_move_from_path(reachedGoal, path, gameState)
        move_eval = path_progress_contribution + self.move_capture_potential(move, gameState)
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
        print('capture potential', end=' ')
        print(nCaptures)

        # find path progress contribution relative to past moves
        gameStateCopy = deepcopy(gameState)
        gameStateCopy.set_cell_color(move, gameStateCopy.player)
        gameStateCopy.nPlayerCells += 1
        self.check_diamond(move[0], move[1], gameStateCopy.player, gameStateCopy)
        path_progress_contribution, _, __ = self.perform_path_search(gameStateCopy)

        # total evaluation of this move
        move_eval = path_progress_contribution + nCaptures
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

    def perform_path_search(self, gameState: Graph):
        cost = inf
        reachedGoal = False
        path: List[Location] = []
        move: Location = []
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
                    path = reconstruct_path(came_from, start, endGoal)

        cost_vector = cost * np.ones(gameState.turn)
        path_cost_dif =  np.subtract(gameState.path_costs[0:gameState.turn],cost_vector)
        gameState.path_costs[gameState.turn-1] = cost
        path_cost_dif = np.flip(path_cost_dif)
        # path_cost_diff is used to find out how much better we are doing than previous moves
        # we now multiply them with a weight from geometric series 1/k^n and get a dot product
        relative_path_cost = np.dot(path_cost_dif, g_weights)
        c1 = 0.4
        c2 = 0
        path_progress_contribution = self.sigmoid(c1,c2,relative_path_cost)*relative_path_cost

        return path_progress_contribution, path, reachedGoal

    def sigmoid(self, c1, c2, x):
        return 1/(1 + np.exp(-c1 * (x - c2)))

    def evaluation_move(self, move):
        self.move_capture_potential(move)

    def Max_Value(self, move, state, alpha, beta, curr_depth, max_depth):
        if curr_depth == max_depth:
            return

    def Min_Value(self, state, ):
        return


    def apply_move(self):

        return

    def getMoves(self):
        (move1, move1_eval)= self.find_capture_move()
