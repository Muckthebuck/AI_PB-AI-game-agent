from path_search import *
from numpy import random

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
        self.begin: List[Location] = []
        self.goal: List[Location] = []
        self.prevMove: Location = []
        self.currMove: Location = []
        self.nlinkedPath = 0
        self.prevLinkedPath = 0
        if player == red:
            for i in range(0, n):
                nCell = [0] * 2
                nCell[0] = 0
                nCell[1] = i
                self.begin.append(tuple(nCell))
                gCell = [0] * 2
                gCell[0] = n - 1
                gCell[1] = i
                self.goal.append(tuple(gCell))
        else:
            self.goal = []
            for i in range(0, n):
                nCell = [0] * 2
                nCell[0] = i
                nCell[1] = 0
                self.begin.append(tuple(nCell))
                gCell = [0] * 2
                gCell[0] = i
                gCell[1] = n - 1
                self.goal.append(tuple(gCell))

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
        self.check_diamond(cell[0], cell[1], player)
        return

    def check_diamond(self, r, c, player):
        to_clear = []
        for cells in self.generate_valid_diamonds(r, c):
            up = self.gameState.cell_color(cells[0])
            down = self.gameState.cell_color(cells[1])
            left = self.gameState.cell_color(cells[2])
            right = self.gameState.cell_color(cells[3])
            if up == down and left == right and up != left:
                if up == player or down == player:
                    to_clear.append(cells[2])
                    to_clear.append(cells[3])
                if left == player or right == player:
                    to_clear.append(cells[0])
                    to_clear.append(cells[1])
        for cell in to_clear:
            self.gameState.set_cell_color(cell, empty)

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
        cost = inf
        reachedGoal = False
        path: List[Location] = []
        move: Location = []
        location: Location = []
        location1: Location = []
        for start in self.begin:
            if self.gameState.cell_color(start) == self.gameState.enemy:
                continue
            reached_goal, came_from, cost_so_far, endGoal = a_star_search(self.gameState, start, self.goal)
            if not reached_goal:
                continue
            else:
                reachedGoal = True
                if cost_so_far[endGoal] < cost:
                    cost = cost_so_far[endGoal]
                    path = reconstruct_path(came_from, start, endGoal)
        print(path)
        if reachedGoal:
            self.prevLinkedPath = self.nlinkedPath
            self.nlinkedPath = 0
            for cell in path:
                if self.gameState.cell_color(cell) == self.color:
                    self.nlinkedPath += 1
                if self.gameState.cell_color(cell) == empty:
                    location = cell
                    break
        location1 = self.find_capture_move()
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

    def find_capture_move(self) -> Location:
        move: Location = []
        nCaptures = 0
        for cell in self.gameState.playerCell:
            r = cell[0]
            c = cell[1]
            cnt = 0
            for cells in self.generate_valid_diamonds(r, c):
                up = self.gameState.cell_color(cells[0])
                down = self.gameState.cell_color(cells[1])
                left = self.gameState.cell_color(cells[2])
                right = self.gameState.cell_color(cells[3])
                if down == empty and left == right and up != left:
                    cnt = self.move_capture_potential(cells[1])
                    if cnt > nCaptures:
                        nCaptures = cnt
                        move = cells[1]
        print('capture potential', end=' ')
        print(nCaptures)
        return move

    def move_capture_potential(self, location: Location):
        r = location[0]
        c = location[1]
        cnt = 0
        for cells in self.generate_valid_diamonds(r, c):
            up = self.gameState.cell_color(cells[0])
            down = self.gameState.cell_color(cells[1])
            left = self.gameState.cell_color(cells[2])
            right = self.gameState.cell_color(cells[3])
            if up == empty and left == right and down != left and down != up:
                cnt += 1
        return cnt
