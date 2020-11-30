from MCS_exploration.gym_ai2thor.envs.mcs_wrapper import McsWrapper
from planner.ff_planner_handler import PlanParser


class McsObjWrapper(McsWrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action, object_id=None, receptacleObjectId=None):
        assert object_id is not None
        object_id = PlanParser.map_legal_object_name_back(object_id)
        if action in ["PickupObject", "OpenObject", "DropObject"]:
            super().step(action=action, objectId=object_id)
        elif action == "PutObjectIntoReceptacle":
            assert receptacleObjectId is not None
            receptacleObjectId = PlanParser.map_legal_object_name_back(receptacleObjectId)
            super().step(action="PutObject", objectId=object_id, receptacleObjectId=receptacleObjectId)

