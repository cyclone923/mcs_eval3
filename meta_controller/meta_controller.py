from tasks.search_object_in_receptacle.face_turner import FaceTurnerResNet
from a3c.task_util import get_action_space_from_names
from gym_ai2thor.envs.mcs_nav import McsNavWrapper
from gym_ai2thor.envs.mcs_face import McsFaceWrapper
from gym_ai2thor.envs.mcs_obj import McsObjWrapper
import os
from planner.ff_planner_handler import PlanParser
from meta_controller.planner_state import GameState
import machine_common_sense
from tasks.bonding_box_navigation_mcs.bonding_box_navigator import BoundingBoxNavigator, SHOW_ANIMATION
#from tasks.bonding_box_navigation_mcs.visibility_road_map import ObstaclePolygon
import matplotlib.pyplot as plt
from MCS_exploration import sequence_generator,main #import SequenceGenerator

MAX_REACH_DISTANCE = 1

def get_goal(goal_string):
    goal = goal_string.split("|")
    _, goal_x, goal_y, goal_z = goal
    goal_x = float(goal_x)
    goal_y = float(goal_y)
    goal_z = float(goal_z)
    return (goal_x, goal_y, goal_z)


class MetaController:
    def __init__(self, env):
        self.env = env
        self.nav_env = McsNavWrapper(env)
        self.face_env = McsFaceWrapper(env)
        self.obj_env = McsObjWrapper(env)

        # self.nav = NavigatorResNet(get_action_space_from_names(self.nav_env.action_names))
        self.nav = BoundingBoxNavigator()
        if isinstance(self.nav, BoundingBoxNavigator):
            self.env.add_obstacle_func = self.nav.add_obstacle_from_step_output
        self.face = FaceTurnerResNet(get_action_space_from_names(self.face_env.action_names))

        self.obstacles = {}

        self.plannerState = None
        self.sequence_generator_object = sequence_generator.SequenceGenerator(None, self.env)


    def plan_on_current_state(self):
        planner = PlanParser(self.plannerState)
        planner.planner_state_to_pddl(self.plannerState)
        return planner.get_plan()

    def get_inital_planner_state(self,scene_config):
        #self.plannerState = GameState(self.env.scene_config)
        self.plannerState = GameState(scene_config)

    def step(self, action_dict, epsd_collector=None):
        assert 'action' in action_dict
        if action_dict['action'] == "GotoLocation":
            goal = get_goal(action_dict['location'])
            current_object_id = None

            for objectId,object_data in self.plannerState.object_loc_info.items():
                #object_data == goal:
                flag = 0
                for k in range(len(goal)):
                    if round(object_data[k],2) != goal[k]:
                        #print (round(object_data[k],2), goal[k])
                        flag = 1
                        break
                if flag == 0 :
                    current_object_id = objectId
                    break

            current_object_loc = self.plannerState.object_loc_info[current_object_id]
            if len(current_object_loc) == 3 :
                final_goal = goal[:]
                success_distance = MAX_REACH_DISTANCE - 0.4
            else :
                final_goal = (float(current_object_loc[3]), float(current_object_loc[4]), float(current_object_loc[5]))
                print("Made a new goal {}".format(final_goal))
                success_distance = 0

            success = self.nav.go_to_goal(
                self.nav_env, final_goal, success_distance)
            if not success:
                print("Navigation Fail")
                return False
            self.plannerState.agent_loc_info[self.plannerState.AGENT_NAME] = goal
            self.plannerState.object_facing = None
        elif action_dict['action'] == "FaceToFront":
            FaceTurnerResNet.look_to_front(self.face_env)
            self.plannerState.face_to_front = True
            self.plannerState.object_facing = None
        elif action_dict['action'] == "FaceToObject":
            goal = get_goal(action_dict['location'])
            FaceTurnerResNet.look_to_direction(self.face_env, goal, epsd_collector)
            object_in_view = [PlanParser.create_legal_object_name(obj.uuid) for obj in self.env.step_output.object_list if not obj.held]
            if action_dict['objectId'] in object_in_view:
                self.plannerState.object_facing = action_dict['objectId']
            else:
                del self.plannerState.object_loc_info[action_dict['objectId']]
                for k in self.plannerState.object_containment_info:
                    if action_dict['objectId'] in self.plannerState.object_containment_info[k]:
                        self.plannerState.object_containment_info[k].remove(action_dict['objectId'])
                if action_dict['objectId'] in self.plannerState.object_open_close_info:
                    del self.plannerState.object_open_close_info[action_dict['objectId']]
                print("Object {} not at {}".format(action_dict['objectId'], action_dict['location']))
            self.plannerState.face_to_front = False
        elif action_dict['action'] == "PickupObject":
            self.obj_env.step("PickupObject", object_id=action_dict['objectId'])
            if self.env.step_output.return_status == "SUCCESSFUL":
                self.plannerState.object_in_hand = action_dict['objectId']
            else:
                print("Pickup {} fail!".format(action_dict['objectId']))
                return False
        elif action_dict['action'] == "LookForObjectInReceptacle":
            found_object = False
            self.env.step(action="MoveAhead", amount=0.8)
            goal = self.plannerState.object_loc_info[action_dict['receptacleId']]
            FaceTurnerResNet.look_to_direction(self.face_env, goal, epsd_collector)
            self.env.step(action="RotateLook", rotation=-45)
            for _ in range(9):
                if action_dict['objectId'] not in [
                    PlanParser.create_legal_object_name(obj.uuid) for obj in self.env.step_output.object_list
                ]:
                    self.face_env.step(action="RotateRight")
                else:
                    self.plannerState.object_facing = action_dict['objectId']
                    for obj in self.env.step_output.object_list:
                        if PlanParser.create_legal_object_name(obj.uuid) == action_dict['objectId']:
                            goal = (obj.position['x'], obj.position['y'], obj.position['z'])
                            FaceTurnerResNet.look_to_direction(self.face_env, goal, epsd_collector)
                    found_object = True
                    break

            if not found_object:
                print("{} not in {}".format(action_dict['objectId'], action_dict['receptacleId']))
                self.plannerState.object_containment_info[action_dict['objectId']].remove(action_dict['receptacleId'])
        elif action_dict['action'] == "PutObjectIntoReceptacle":
            self.obj_env.step(
                "PutObjectIntoReceptacle", object_id=action_dict['objectId'],
                receptacleObjectId=action_dict['receptacleId']
            )
            if self.env.step_output.return_status == "SUCCESSFUL":
                self.plannerState.object_in_hand = None
                self.plannerState.object_containment_info[action_dict['objectId']] = action_dict['receptacleId']
            else:
                self.plannerState.knowledge.canNotPutin.add((action_dict['objectId'], action_dict['receptacleId']))
        elif action_dict['action'] == "OpenObject":
            self.obj_env.step("OpenObject", object_id=action_dict['objectId'])
            if self.env.step_output.return_status == "SUCCESSFUL":
                self.plannerState.object_open_close_info[action_dict['objectId']] = True
            else:
                print("Open {} fail, {}".format(action_dict['objectId'], self.env.step_output.return_status))
                return False
        elif action_dict['action'] == "DropObjectNextTo":
            FaceTurnerResNet.look_to_front(self.face_env)
            self.obj_env.step("DropObject", object_id=action_dict['objectId'])
            if self.env.step_output.return_status == "SUCCESSFUL":
                self.plannerState.knowledge.objectNextTo[action_dict['objectId']] = action_dict['goal_objectId']
                self.plannerState.object_in_hand = None
                print("Drop Successful, if reward 0 then it should be out of range.")

            '''
            for _ in range(10):
                print("Head_tilt before drop object {}".format(self.env.step_output.head_tilt))
                self.obj_env.step("DropObject", object_id=action_dict['objectId'], epsd_collector=epsd_collector)
                if self.env.step_output.return_status == "SUCCESSFUL":
                    self.plannerState.knowledge.objectNextTo[action_dict['objectId']] = action_dict['goal_objectId']
                    self.plannerState.object_in_hand = None
                    print("Drop Successful, if reward 0 then it should be out of range.")
                    break
                self.env.step(action="MoveBack", amount=0.05)
            '''
            if self.plannerState.object_in_hand:
                print("DropObject fail")
                return False

        elif action_dict['action'] == "DropObjectOnTopOf":
            self.obj_env.step(
                "PutObjectIntoReceptacle", object_id=action_dict['objectId'],
                receptacleObjectId=action_dict['goal_objectId']
            ) # use magical action at current point
            if self.env.step_output.return_status == "SUCCESSFUL":
                self.plannerState.knowledge.objectOnTopOf[action_dict['objectId']] = action_dict['goal_objectId']
                self.plannerState.object_in_hand = None
                print("PutObject succeeded! If reward 0 then it is a bug!")
                # assert self.env.step_output.reward == 1
            else:
                print("PutObjectIntoReceptacle {}".format(self.env.step_output.return_status))
                return False
        return True

    def excecute(self):
        scene_config = main.explore_scene(self.sequence_generator_object, self.env.step_output)#'retrieval-', '0001'
        self.get_inital_planner_state(scene_config)
        if isinstance(self.nav, BoundingBoxNavigator):
            self.nav.clear_obstacle_dict()
            #self.nav.initialize_scene_dict(scene_config['obstacles'])
            self.nav.scene_obstacles_dict = scene_config['obstacles']
        meta_stage = 0
        while True:
            # print("Meta-Stage: {}".format(meta_stage))
            result_plan = self.plan_on_current_state()
            # print(result_plan[0])
            success = self.step(result_plan[0])
            if not success:
                break
            if result_plan[0]['action'] == "End":
                break
            meta_stage += 1
        # time.sleep(2)
        print("Task Reward: {}\n".format(self.env.step_output.reward))
        return True
