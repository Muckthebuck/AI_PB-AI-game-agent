from typing import Dict, List, Tuple, TypeVar, Optional
import numpy as np
from numpy import inf

T = TypeVar('T')
Location = TypeVar('Location')
Color = TypeVar('Color')
red = "red"
blue = "blue"
empty = "empty"


class Graph:
    """
        adapted from https://www.redblobgames.com/pathfinding/a-star/implementation.html
    """

    def __init__(self, n, player):
        self.player = player
        self.enemy = red if player == blue else blue
        self.cell: Dict[Location, List[Color, List[Location]]] = {}
        self.nlinkedPath = 0
        self.prevLinkedPath = 0
        self.blue_turn = 0
        self.red_turn = 0
        self.maxTurns = 344
        self.nPlayerCells = 0
        self.nEnemyCells = 0
        self.begin: List[Location] = []
        self.goal: List[Location] = []
        self.red_begin: List[Location] = []
        self.red_goal: List[Location] = []
        self.blue_begin: List[Location] = []
        self.blue_goal: List[Location] = []
        self.red_cells: Dict[Location, Color] = {}
        self.blue_cells: Dict[Location, Color] = {}
        self.red_path_costs = n*np.ones(self.maxTurns)
        self.blue_path_costs = n*np.ones(self.maxTurns)
        moves = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]]
        for i in range(0, n):
            for j in range(0, n):
                curr = [i, j]
                neighbours = []
                color = "empty"
                for k in moves:
                    nCell = [0] * 2
                    nCell[0] = curr[0] + k[0]
                    nCell[1] = curr[1] + k[1]
                    if not (nCell[0] < 0 or nCell[0] >= n or nCell[1] < 0 or nCell[1] >= n):
                        neighbours.append(tuple(nCell))
                self.cell.update({tuple(curr): [color, neighbours]})
                # print(curr, neighbours)
        for i in range(0, n):
            nCell = [0] * 2
            nCell[0] = 0
            nCell[1] = i
            self.red_begin.append(tuple(nCell))
            gCell = [0] * 2
            gCell[0] = n - 1
            gCell[1] = i
            self.red_goal.append(tuple(gCell))
        for i in range(0, n):
            nCell = [0] * 2
            nCell[0] = i
            nCell[1] = 0
            self.blue_begin.append(tuple(nCell))
            gCell = [0] * 2
            gCell[0] = i
            gCell[1] = n - 1
            self.blue_goal.append(tuple(gCell))
        if player == red:
            self.begin = self.red_begin
            self.goal = self.red_goal
        else:
            self.begin = self.blue_begin
            self.goal = self.blue_goal

    def flip_player_color(self):
        self.player, self.enemy = self.enemy, self.player
        if self.player == red:
            self.begin = self.red_begin
            self.goal = self.red_goal
        else:
            self.begin = self.blue_begin
            self.goal = self.blue_goal

    def print(self):
        for cell, neighbours in self.cell.items():
            print(cell, neighbours)

    def neighbors(self, location: Location) -> List[Location]:
        return self.cell.get(location)[1]

    def cell_color(self, location: Location) -> Color:
        return self.cell.get(location)[0]

    def set_cell_color(self, location: Location, color: Color):
        if self.cell_color(location) == self.player and color != self.player:
            # if the current id is no longer a player colour
            if self.player == blue:
                self.blue_cells.pop(location)
            elif self.player == red:
                self.red_cells.pop(location)
        elif color == self.player:
            # if the current id will be a player
            if self.player == blue:
                self.blue_cells.update({location: color})
            elif self.player == red:
                self.red_cells.update({location: color})
        elif self.cell_color(location) == self.enemy and color != self.enemy:
            # if the current id is no longer an enemy colour
            if self.enemy == blue:
                self.blue_cells.pop(location)
            elif self.enemy == red:
                self.red_cells.pop(location)
        elif color == self.enemy:
            # if the current id will be an enemy
            if self.enemy == blue:
                self.blue_cells.update({location: color})
            elif self.enemy == red:
                self.red_cells.update({location: color})
        self.cell[location][0] = color

    def get_player_cells(self) -> Dict[Location, Color]:
        if self.player == blue:
            return self.blue_cells
        elif self.player == red:
            return self.red_cells

    def get_enemy_cells(self) -> Dict[Location, Color]:
        if self.player == blue:
            return self.red_cells
        elif self.player == red:
            return self.blue_cells

    def get_player_bounds(self) -> Tuple[List[Location], List[Location]]:
        if self.player == blue:
            return self.blue_begin, self.blue_goal
        elif self.player == red:
            return self.red_begin, self.red_goal

    def get_enemy_bounds(self) -> Tuple[List[Location], List[Location]]:
        if self.player == red:
            return self.blue_begin, self.blue_goal
        elif self.player == blue:
            return self.red_begin, self.red_goal

    def cost(self, src: Location, dst: Location) -> float:
        pass

    def reconstruct_path(self, came_from: Dict[Location, Location],
                         start: Location, goal: Location) -> List[Location]:
        """
            adapted from https://www.redblobgames.com/pathfinding/a-star/implementation.html
        """
        current: Location = goal
        path: List[Location] = []
        while current != start:  # note: this will fail if no path found
            if self.cell_color(current) != empty:
                current = came_from[current]
                continue
            path.append(current)
            current = came_from[current]
        if self.cell_color(start) == empty:
            path.append(start)
        path.reverse()  # optional
        return path


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def empty(self) -> bool:
        return not self.elements

    def getParent(self, key) -> int:
        if key <= 2:
            return 0
        else:
            return (key - 1) // 2

    def min_heapify(self, key):
        # not used rn
        left = (2 * key) + 1
        right = (2 * key) + 2
        if not right >= len(self.elements):
            node = self.elements[key]
            if node[0] > (self.elements[left])[0] or node[0] > self.elements[right][0]:
                if (self.elements[right])[0] > (self.elements[left])[0]:
                    self.elements[key], self.elements[left] = self.elements[left], self.elements[key]
                    self.min_heapify(left)
                else:
                    self.elements[key], self.elements[right] = self.elements[right], self.elements[key]
                    self.min_heapify(right)

    def push(self, item: T, priority: float):
        self.elements.append((priority, item))
        curr = len(self.elements) - 1
        while self.elements[curr][0] < self.elements[self.getParent(curr)][0]:
            self.elements[curr], self.elements[self.getParent(curr)] = self.elements[self.getParent(curr)], \
                                                                       self.elements[curr]
            curr = self.getParent(curr)

    def pop(self) -> T:
        head = self.elements[0]
        self.elements[0] = self.elements[-1]
        self.elements.pop()
        if len(self.elements) - 1 > 0:
            self.min_heapify(0)
        return head[1]
        # return heapq.heappop(self.elements)[1]


def reconstruct_path(came_from: Dict[Location, Location],
                     start: Location, goal: Location) -> List[Location]:
    """
        adapted from https://www.redblobgames.com/pathfinding/a-star/implementation.html
    """
    current: Location = goal
    path: List[Location] = []
    while current != start:  # note: this will fail if no path found
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional
    return path



def find_dist(H1, H2):
    """
    # adapted from redblobgames.com/grids/hexagon
    Args:
        H1: src hexagon cell coordinate tuple
        H2: dst hexagon cell coordinate tuple

    #Relative direction, i.e. which wedge side i need to go towards
    1  .-'-. 0
    2 |     | 5
    3 '-._.-'4
    Returns:(manhattan distance, relative direction H2 to H1)

    """

    r = H1[0] - H2[0]
    q = H1[1] - H2[1]
    s = -r - q
    d = (abs(r) + abs(q) + abs(s)) / 2

    return d


def find_dist2(H1, goalList):
    dist = inf
    if not goalList:
        return dist
    for H2 in goalList:
        cost = find_dist(H1, H2)
        if cost < dist:
            dist = cost

    return dist


def a_star_search(graph: Graph, start: Location, goal: List[Location]):
    """
        adapted from https://www.redblobgames.com/pathfinding/a-star/implementation.html
    """
    frontier = PriorityQueue()
    frontier.push(start, 0)
    came_from: Dict[Location, Optional[Location]] = {}
    cost_so_far: Dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0 if graph.cell_color(start) == graph.player else 1
    reached_goal = False
    endGoal: Location = []
    while not frontier.empty():
        current: Location = frontier.pop()

        if current in goal:
            endGoal = current
            reached_goal = True
            break

        for next in graph.neighbors(current):
            if graph.cell_color(next) == graph.enemy:
                # dont want to consider enemy occupied space
                continue
            move_cost = 0 if graph.cell_color(next) == graph.player else 1
            new_cost = cost_so_far[current] + move_cost  # +1 for cost to next node
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + find_dist2(next, goal)
                frontier.push(next, priority)
                came_from[next] = current

    return reached_goal, came_from, cost_so_far, endGoal
