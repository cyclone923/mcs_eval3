from gym_ai2thor.envs.mcs_env import McsEnv
from frame_collector import Frame_collector
import numpy as np
import sys
import json

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)

# def my_floor(a, precision=0):
#     return np.round(a - 0.5 * 10**(-precision), precision)

class McsHumanControlEnv(McsEnv):
    def __init__(self,  **args):
        super().__init__(**args)

    def step(self, action_str, **args):
        print("Action you entered: {} {}".format(action_str, args))
        if "Move" in action_str:
            self.step_output = super().step(action=action_str)
        elif "Look" in action_str:
            if action_str == "LookUp":
                self.step_output = super().step(action="LookUp", **args)
            elif action_str == "LookDown":
                self.step_output = super().step(action="LookDown", **args)
            else:
                raise NotImplementedError
        elif "Rotate" in action_str:
            if action_str == "RotateLeft":
                self.step_output = super().step(action="RotateLeft", **args)
            elif action_str == "RotateRight":
                self.step_output = super().step(action="RotateRight", **args)
            else:
                self.step_output = super().step(action="RotateObject", **args)
        elif action_str == "PickupObject":
            self.step_output = super().step(action="PickupObject", **args)
            print("PickObject {}".format(self.step_output.return_status))
        elif action_str == "PutObject":
            self.step_output = super().step(action="PutObject", **args)
            print("PutObject {}".format(self.step_output.return_status))
        elif action_str == "DropObject":
            self.step_output = super().step(action="DropObject", **args)
            print("DropObject {}".format(self.step_output.return_status))
        elif action_str == "ThrowObject":
            self.step_output = super().step(action="ThrowObject", **args)
            print("ThrowObject {}".format(self.step_output.return_status))
        elif action_str == "PushObject":
            self.step_output = super().step(action="PushObject", **args)
        elif action_str == "PullObject":
            self.step_output = super().step(action="PullObject", **args)
        elif action_str == "OpenObject":
            self.step_output = super().step(action="OpenObject", **args)
        elif action_str == "CloseObject":
            self.step_output = super().step(action="CloseObject", **args)
        else:
            self.step_output = super().step(action=action_str)

    def print_step_output(self, save_radius=True):
        print("- " * 20)
        depth_img = np.array(self.step_output.depth_map_list[0])
        print("Previous Action Status: {}".format(self.step_output.return_status))
        if hasattr(self.step_output, "reward"):
            print("Previous Reward: {}".format(self.step_output.reward))
        p = {'x': None, 'y': None, 'z': None}
        if self.step_output.position is not None:
            p = self.step_output.position
        print(
            "Agent at: ({}, {}, {}), HeadTilt: {:.2f}, Rotation: {}".format(
                p['x'],
                p['y'],
                p['z'],
                self.step_output.head_tilt,
                self.step_output.rotation
            )
        )
        print("Camera Field of view {}".format(self.step_output.camera_field_of_view))
        print("Visible Objects:")
        all_radius = {}
        for obj in self.step_output.object_list:
            print("Distance {:.3f} to {} {} {} ({:.3f},{:.3f},{:.3f}) with {} bdbox points".format(
                obj.distance_in_world, obj.visible, obj.shape, obj.uuid,
                obj.position['x'], obj.position['y'], obj.position['z'], len(obj.dimensions))
            )
            radius_info = self.print_object_size(obj.dimensions)
            all_radius[obj.uuid] = radius_info
        if save_radius:
            with open('interaction_scenes/box_trophy_size.json', 'w') as fp:
                json.dump(all_radius, fp, indent=4)

        for obj in self.step_output.structural_object_list:
            if "wall" not in obj.uuid:
                continue
            print("Distance {:.3f} to {} {} {} ({:.3f},{:.3f},{:.3f}) with {} bdbox points".format(
                obj.distance_in_world, obj.visible, obj.shape, obj.uuid,
                obj.position['x'], obj.position['y'], obj.position['z'], len(obj.dimensions))
            )

        print("{} depth images returned".format(len(self.step_output.depth_map_list)))
        print("{} object images returned".format(len(self.step_output.object_mask_list)))
        print("{} scene images returned".format(len(self.step_output.image_list)))
        print("{} objects' properties".format(len(self.step_output.object_list)))

    def print_object_size(self, dimensions):
        x_range = [np.inf, -np.inf]
        y_range = [np.inf, -np.inf]
        z_range = [np.inf, -np.inf]
        for p in dimensions:
            x_range[0] = min(x_range[0], p['x'])
            y_range[0] = min(y_range[0], p['y'])
            z_range[0] = min(z_range[0], p['z'])
            x_range[1] = max(x_range[1], p['x'])
            y_range[1] = max(y_range[1], p['y'])
            z_range[1] = max(z_range[1], p['z'])

        x_radius = (x_range[1] - x_range[0]) / 2
        y_radius = (y_range[1] - y_range[0]) / 2
        z_radius = (z_range[1] - z_range[0]) / 2

        # print("x_r: {}".format(my_ceil(x_radius, 2)))
        # print("y_r: {}".format(my_ceil(y_radius, 2)))
        # print("z_r: {}".format(my_ceil(z_radius, 2)))

        info = {
            "x_r": my_ceil(x_radius, 2),
            "y_r": my_ceil(y_radius, 2),
            "z_r": my_ceil(z_radius, 2),
        }
        print(info)
        return info





if __name__ == '__main__':
    start_scene_number = 0
    collector = Frame_collector(scene_dir="simple_task_img", start_scene_number=start_scene_number)
    env = McsHumanControlEnv(task="interaction_scenes", scene_type="eval3", start_scene_number=start_scene_number, frame_collector=collector)
    env.reset()

    while True:
        action = input("Enter Action: ")
        if action == "w":
            env.step("MoveAhead")
        elif action == "s":
            env.step("MoveBack")
        elif action == "a":
            env.step("MoveLeft")
        elif action == "d":
            env.step("MoveRight")
        elif action == "q":
            # rt = input("RotateLeft! Enter the rotation: ")
            rt = 10
            env.step("RotateLeft", rotation=-float(rt))
        elif action == "e":
            # rt = input("RotateRight! Enter the rotation: ")
            rt = 10
            env.step("RotateRight", rotation=float(rt))
        elif action == "r":
            # hrz = input("Look Up! Enter the horizon: ")
            hrz = 10
            env.step("LookUp", horizon=-float(hrz))
        elif action == "f":
            # hrz = input("Look Down! Enter the horizon: ")
            hrz = 10
            env.step("LookDown", horizon=float(hrz))
        elif action == "U":
            x, y = input("Pickup Object! Enter the object x and y coord on 2D image separated by space:").split()
            env.step("PickupObject", objectImageCoordsX=int(x),  objectImageCoordsY=int(y))
        elif action == "I":
            x, y = input("Put Object! Enter the receptacle x and y coord on 2D image separated by space:").split()
            env.step("PutObject", objectImageCoordsX=int(x),  objectImageCoordsY=int(y))
        elif action == "O":
            print("Drop Object!")
            env.step("DropObject")
        elif action == "P":
            print("Throw Object!")
            force = input("Throw Object! Enter the force: ")
            env.step("ThrowObject", force=int(force), objectDirectionY=env.step_output.head_tilt)
        elif action == "J":
            x, y = input("Push Object! Enter the object x and y coord on 2D image separated by space:").split()
            env.step("PushObject", objectImageCoordsX=int(x),  objectImageCoordsY=int(y))
        elif action == "K":
            x, y = input("Pull Object! Enter the object x and y coord on 2D image separated by space:").split()
            env.step("PullObject", objectImageCoordsX=int(x),  objectImageCoordsY=int(y))
        elif action == "L":
            x, y = input("Rotate Object! Enter the object x and y coord on 2D image separated by space:").split()
            env.step("RotateObject", objectImageCoordsX=int(x),  objectImageCoordsY=int(y), rotationY=10)
        elif action == "N":
            x, y = input("Open Object! Enter the object x and y coord on 2D image separated by space:").split()
            env.step("OpenObject", objectImageCoordsX=int(x),  objectImageCoordsY=int(y))
        elif action == "M":
            x, y = input("Close Object! Enter the object x and y coord on 2D image separated by space:").split()
            env.step("CloseObject", objectImageCoordsX=int(x),  objectImageCoordsY=int(y))
        elif action == "z":
            break
        elif action == 'p':
            env.print_step_output()
            print("- " * 10)
        else:
            print("Invalid Action")