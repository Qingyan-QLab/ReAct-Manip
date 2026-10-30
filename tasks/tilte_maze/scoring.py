"""Continuous route progress and yellow-zone penalties for the maze."""

from collections import deque
import math

from .assets.gen_assets import GRID_SIZE, MAZE_GRID

SCORE_MAX = 100.0
YELLOW_ENTRY_PENALTY = 20.0
MAX_PENALIZED_ENTRIES = 4


def target_route(board_half_size_m=0.17, rim_width_m=0.012, ball_radius_m=0.0122):
    """Grey-cell route whose centers the enlarged ball can physically reach."""
    green = next((x, y) for y, row in enumerate(MAZE_GRID)
                 for x, kind in enumerate(row) if kind == "s")
    # The ball overlaps the green tile but its center starts one row below it
    # so the sphere does not intersect the top rim.
    start = green[0], green[1] + 1
    cell_size = 2 * board_half_size_m / GRID_SIZE
    rim_cells = rim_width_m / cell_size
    required_clearance = ball_radius_m / cell_size + 0.01
    walls = [(x, y) for y, row in enumerate(MAZE_GRID)
             for x, kind in enumerate(row) if kind in "#r"]

    def center_is_clear(x, y):
        cx, cy = x + 0.5, y + 0.5
        if min(cx - rim_cells, cy - rim_cells,
               GRID_SIZE - rim_cells - cx, GRID_SIZE - rim_cells - cy) < required_clearance:
            return False
        for wx, wy in walls:
            dx = max(wx - cx, cx - wx - 1, 0.0)
            dy = max(wy - cy, cy - wy - 1, 0.0)
            if dx * dx + dy * dy < required_clearance * required_clearance:
                return False
        return True

    safe = {(x, y) for y, row in enumerate(MAZE_GRID)
            for x, kind in enumerate(row) if kind in "sgG" and center_is_clear(x, y)}
    if start not in safe:
        raise ValueError("The enlarged ball has no collision-free start cell")
    queue = deque([start])
    previous = {start: None}
    goal = None
    while queue:
        cell = queue.popleft()
        if MAZE_GRID[cell[1]][cell[0]] == "G":
            goal = cell
            break
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            x, y = neighbor
            if neighbor in safe and neighbor not in previous:
                previous[neighbor] = cell
                queue.append(neighbor)
    if goal is None:
        raise ValueError("The grey maze route does not reach a blue goal cell")
    route = []
    while goal is not None:
        route.append(goal)
        goal = previous[goal]
    return tuple(reversed(route))


ROUTE = target_route()


class MazeScore:
    """Track route progress and penalize each entry into a yellow region."""

    def __init__(self, board_half_size_m: float, rim_width_m: float = 0.012,
                 ball_radius_m: float = 0.0122):
        self.half = float(board_half_size_m)
        self.cell_size = 2 * self.half / GRID_SIZE
        self.route = target_route(self.half, rim_width_m, ball_radius_m)
        self.route_index = {cell: index for index, cell in enumerate(self.route)}
        self.reset()

    def reset(self):
        self.max_progress = 0.0
        self.yellow_entries = 0
        self.yellow_active = False
        self.success = False
        self.score = 0.0

    @property
    def progress_pct(self):
        return SCORE_MAX * self.max_progress / (len(self.route) - 1)

    @property
    def penalty_points(self):
        return YELLOW_ENTRY_PENALTY * min(self.yellow_entries, MAX_PENALIZED_ENTRIES)

    def cell_at(self, xy):
        x = math.floor((float(xy[0]) + self.half) / self.cell_size)
        y = math.floor((self.half - float(xy[1])) / self.cell_size)
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            return x, y
        return None

    def _center(self, cell):
        x, y = cell
        return (-self.half + (x + 0.5) * self.cell_size,
                self.half - (y + 0.5) * self.cell_size)

    def _progress_at(self, xy, cell):
        index = self.route_index.get(cell)
        if index is None:
            return None
        if index == 0:
            return 0.0
        point = float(xy[0]), float(xy[1])
        candidates = []
        for segment in (index - 1, index):
            if not 0 <= segment < len(self.route) - 1:
                continue
            ax, ay = self._center(self.route[segment])
            bx, by = self._center(self.route[segment + 1])
            dx, dy = bx - ax, by - ay
            t = min(1.0, max(0.0, ((point[0] - ax) * dx + (point[1] - ay) * dy)
                                  / (dx * dx + dy * dy)))
            distance_sq = (point[0] - ax - t * dx) ** 2 + (point[1] - ay - t * dy) ** 2
            candidates.append((distance_sq, segment + t))
        return min(candidates)[1] if candidates else float(index)

    def update(self, xy):
        """Return this observation's score delta and newly triggered events."""
        if self.success:
            return 0.0, []
        old_score = self.score
        cell = self.cell_at(xy)
        kind = MAZE_GRID[cell[1]][cell[0]] if cell is not None else None
        events = []
        in_yellow = kind == "t"
        if in_yellow and not self.yellow_active:
            self.yellow_entries += 1
            events.append("ball_entered_trap")
        self.yellow_active = in_yellow
        if cell is not None:
            progress = self._progress_at(xy, cell)
            if progress is not None:
                self.max_progress = max(self.max_progress, progress)
        if kind == "G":
            self.max_progress = float(len(self.route) - 1)
            self.success = True
            events.append("ball_reached_goal")
        self.score = min(SCORE_MAX, max(0.0, self.progress_pct - self.penalty_points))
        return self.score - old_score, events
