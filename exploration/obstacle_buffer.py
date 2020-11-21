from exploration.roadmap.visibility_road_map import ObstaclePolygon
from exploration.util import pre_process, depth_to_points
import numpy as np
import sys
from vision.instSeg.inference import MaskAndClassPredictor
import matplotlib.pyplot as plt
from vision.instSeg.data.config_mcsVideo3_inter import MCSVIDEO_INTER_CLASSES_BG, MCSVIDEO_INTER_CLASSES_FG

TROPHY_INDEX = MCSVIDEO_INTER_CLASSES_FG.index('trophy') + 1
BOX_INDEX = MCSVIDEO_INTER_CLASSES_FG.index('box') + 1

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


class OccupancyMap:

    GRID_SIZE = 0.1
    PADDING = 10
    H = 400
    OFFSET_H = H // 2
    W = 400
    OFFSET_W = W // 2

    def __init__(self):
        self.map = None

    def reset(self):
        self.map = np.zeros(shape=(self.H, self.W))

    def update(self, pts):
        pts = np.round(pts, decimals=1)
        pts[:, 0] = (pts[:, 0] / self.GRID_SIZE) + self.OFFSET_H
        pts[:, 1] = (pts[:, 1] / self.GRID_SIZE) + self.OFFSET_W
        pts = pts.astype(int)
        pts = np.unique(pts, axis=0)
        rows, cols = zip(*pts)
        self.map[rows, cols] = 1
        print(self.map)
        a = 1

    def get_obstacles(self):
        return

class ObstacleBuffer:

    def __init__(self, level):
        self.level = level
        self.scene_obstacles_dict = None
        if self.level != 'oracle':
            self.occupancy_map = OccupancyMap()
            self.mask_predictor = MaskAndClassPredictor(cuda=False)

    def reset(self):
        self.scene_obstacles_dict = {}
        if self.level != 'oracle':
            self.occupancy_map.reset()

    @property
    def scene_obstacles(self):
        return self.scene_obstacles_dict

    def add_obstacle_oracle(self, agent):
        step_output = agent.step_output
        for obj in step_output.object_list:
            if len(obj.dimensions) > 0 and obj.uuid not in self.scene_obstacles_dict and obj.visible:
                x_list = []
                y_list = []
                for i in range(4, 8):
                    x_list.append(obj.dimensions[i]['x'])
                    y_list.append(obj.dimensions[i]['z'])
                self.scene_obstacles_dict[obj.uuid] = ObstaclePolygon(x_list, y_list)
            if obj.held:
                del self.scene_obstacles_dict[obj.uuid]

        for obj in step_output.structural_object_list:
            if len(obj.dimensions) > 0 and obj.uuid not in self.scene_obstacles_dict and obj.visible:
                if obj.uuid == "ceiling" or obj.uuid == "floor":
                    continue
                x_list = []
                y_list = []
                for i in range(4, 8):
                    x_list.append(obj.dimensions[i]['x'])
                    y_list.append(obj.dimensions[i]['z'])
                self.scene_obstacles_dict[obj.uuid] = ObstaclePolygon(x_list, y_list)

    def add_obstacle_level1(self, agent):
        # out = self.prediction_level1(agent)
        self.add_obstacle_oracle(agent)
        # pts = depth_to_points(
        #     agent.depth_map, agent.agent_fov,
        #     agent.step_output.position, agent.agent_rotation,
        #     agent.agent_head_tilt
        # )
        # pts = pre_process(pts)
        # self.occupancy_map.update(pts)

    def prediction_level1(self, agent):
        rgbI = np.array(agent.image_map)
        bgrI = rgbI[:, :, [2,1,0]]
        depthI = np.uint8(agent.depth_map / agent.step_output.camera_clipping_planes[1] * 255)

        ret = self.mask_predictor.step(bgrI, depthI)
        cls = np.argmax(ret['obj_class_score'], axis=1)
        print(cls)
        if TROPHY_INDEX in cls:
            n_th_obj = np.where(cls == TROPHY_INDEX)[0]
            print("find trophy in {}th fg object".format(n_th_obj))

        # self.debug_out(bgrI, depthI, ret)
        a = 1


    def debug_out(self, bgrI, depthI, ret):
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(bgrI[..., [2, 1, 0]])
        ax[0, 0].set_title('RGB image')
        ax[0, 1].imshow(depthI, cmap='gray')
        ax[0, 1].set_title('depth image')
        ax[1, 0].imshow(ret['net-mask'])
        ax[1, 0].set_title('net predict mask')
        ax[1, 1].imshow(ret['mask_prob'].argmax(axis=0))
        ax[1, 1].set_title('final mask (with cls-score)')
        plt.show()
