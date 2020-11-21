import machine_common_sense
import shapely.geometry.polygon as sp
import math
from exploration.roadmap.fov import FieldOfView

MOVE_DISTANCE = machine_common_sense.controller.MOVE_DISTANCE
SCENE_X_RANGE = [-5, 5]
SCENE_Z_RANGE = [-5, 5]
ROUND_SCENE_X_RANGE = [int(i / MOVE_DISTANCE) for i in SCENE_X_RANGE]
ROUND_SCENE_Z_RANGE = [int(i / MOVE_DISTANCE) for i in SCENE_Z_RANGE]
COVER_FLOOR_SAMPLE_STEP_SIZE = int(1 / MOVE_DISTANCE)

OVERALL_AREA = (SCENE_X_RANGE[1] - SCENE_X_RANGE[0]) * (SCENE_Z_RANGE[1] - SCENE_Z_RANGE[0])




class ExploreStrategy:

    def __init__(self, agent):
        self.agent = agent
        self.world_poly = None
        self.visited_cell = set()
        self.current_view = None

    def reset(self):
        self.world_poly = sp.Polygon()
        self.visited_cell = set()

    def update_state(self):
        self.current_view = self.get_coverage(
            self.agent.agent_x, self.agent.agent_z,
            self.agent.agent_rotation, self.agent.agent_fov_radian,
            self.agent.scene_obstacles.values()
        )
        self.world_poly = self.world_poly.union(self.current_view)

    def get_coverage(self, x, z, rotation, camera_fov_rad, obstacles):
        rotation_rad = rotation / 180.0 * math.pi
        fov_checker = FieldOfView([x, z, rotation_rad], camera_fov_rad, obstacles)
        checkPoly = fov_checker.getFoVPolygon(17)

        intersection_free_points = [[], []]
        for i, j in zip(checkPoly.x_list, checkPoly.y_list):
            if len(intersection_free_points[0]) != 0:
                if not (abs(i - intersection_free_points[0][-1]) < 0.001 and abs(
                        j - intersection_free_points[1][-1]) < 0.001):
                    intersection_free_points[0].append(i)
                    intersection_free_points[1].append(j)
            else:
                intersection_free_points[0].append(i)
                intersection_free_points[1].append(j)
        view = sp.Polygon(zip(intersection_free_points[0], intersection_free_points[1]))
        return view

    def initial_exploration(self):
        look_around = ['RotateRight' for _ in range(36)]
        actions = ['LookDown' for _ in range(2)] + look_around + ['LookUp' for _ in range(2)]
        actions = look_around
        for a in actions:
            self.agent.step(action=a)
            if self.agent.goal_found:
                break

    def find_next_best_pose(self):

        def check_goal_valid(x, z, obstacles):
            p = sp.Point(x, z)
            valid = True
            for poly in obstacles:
                if p.within(poly):
                    valid = False
                    break
            return valid

        best_area = 0
        best_poly = None
        best_x, best_z, best_r = 0, 0, 0
        exploration_routine = self.flood_fill(self.agent.agent_x, self.agent.agent_z)
        for x, z in exploration_routine:
            if not check_goal_valid(x, z, self.agent.scene_obstacles.values()):
                continue
            for r in [self.agent.agent_rotation + i * 60 for i in range(6)]:
                if r > 360:
                    r -= 360
                try:
                    poly = self.get_coverage(x, z, r, self.agent.agent_fov_radian, self.agent.scene_obstacles.values())
                except:
                    continue
                newPoly = poly.difference(self.world_poly)
                if newPoly.area > best_area:
                    best_area = newPoly.area
                    best_x, best_z, best_r = x, z, r
                    best_poly = newPoly

        self.visited_cell.add((best_x, best_z))
        return best_x, best_z, best_r / 360 * 2 * math.pi

    def flood_fill(self, x, y, max_depth=5):
        x = round(x / MOVE_DISTANCE)
        y = round(y / MOVE_DISTANCE)

        curr_q = []
        covered = []
        covered.append((x, y))
        curr_q.append((x, y, 0))
        while curr_q:
            x1, y1, depth = curr_q.pop()
            if depth > max_depth:
                continue
            for step in [-1, 1]:
                next_x = x1 + step * COVER_FLOOR_SAMPLE_STEP_SIZE
                next_y = y1

                if self.check_validity(next_x, next_y, covered):
                    covered.append((next_x, next_y))
                    curr_q.append((next_x, next_y, depth+1))

                next_x = x1
                next_y = y1 + step * COVER_FLOOR_SAMPLE_STEP_SIZE
                if self.check_validity(next_x, next_y, covered):
                    covered.append((next_x, next_y))
                    curr_q.append((next_x, next_y, depth+1))

        return [(round(i[0] * MOVE_DISTANCE, 1), round(i[1] * MOVE_DISTANCE)) for i in covered]

    def check_validity(self, x, z, q):
        ans = False
        if not x <= ROUND_SCENE_X_RANGE[0]:
            if not x >= ROUND_SCENE_X_RANGE[1]:
                if not z <= ROUND_SCENE_Z_RANGE[0]:
                    if not z >= ROUND_SCENE_Z_RANGE[1]:
                        if not ((x, z)) in q:
                            if not ((x, z)) in self.visited_cell:
                                ans = True
        return ans










