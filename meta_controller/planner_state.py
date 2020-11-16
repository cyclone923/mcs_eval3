from planner.ff_planner_handler import PlanParser
from collections import defaultdict


class GameState:

    AGENT_NAME = "agent1"
    AGENT_Y_INIT = 0.4625

    def __init__(self, config):
        self.face_to_front = False
        self.object_facing = None
        self.object_in_hand = None
        self.knowledge = GameKnowlege()

        self.agent_loc_info = {
            self.AGENT_NAME: (
                config['performerStart']['position']['x'],
                self.AGENT_Y_INIT,
                config['performerStart']['position']['z']
            )
        }
        self.object_loc_info = {}
        self.object_open_close_info = {}
        for obj in config['objects']:
            object_id = PlanParser.create_legal_object_name(obj['uuid'])
            if 'agent_position' not in obj:
                self.object_loc_info[object_id] = (
                    obj['position']['x'],
                    obj['position']['y'],
                    obj['position']['z']
                )
            else :
                self.object_loc_info[object_id] = (
                    obj['position']['x'],
                    obj['position']['y'],
                    obj['position']['z'],
                    obj['agent_position']['x'],
                    obj['agent_position']['y'],
                    obj['agent_position']['z']
                )
            #self.object_loc_info[object_id] =
            if "opened" in obj:
                self.object_open_close_info[object_id] = obj["opened"]

        self.object_containment_info = defaultdict(lambda : [])
        self.object_knowledge_info = {}

        self.goal_predicate_list = None
        self.goal_category = None
        self.creat_goal(config)

    def creat_goal(self, config):
        if 'goal' in config:
            self.goal_category = config['goal']['category']
            self.goal_predicate_list = []
            if self.goal_category == "traversal":
                self.goal_object_id = PlanParser.create_legal_object_name(config['goal']['metadata']['target']['id'])
                if self.goal_object_id not in self.object_loc_info:
                    return
                agent_final_loc = PlanParser.replace_digital_number(
                    "loc|{:.2f}|{:.2f}|{:.2f}".format(
                        self.object_loc_info[self.goal_object_id][0],
                        self.object_loc_info[self.goal_object_id][1],
                        self.object_loc_info[self.goal_object_id][2]
                    )
                )
                self.goal_predicate_list.append(
                    "(agentAtLocation {} {})".format(self.AGENT_NAME, agent_final_loc)
                )
            elif self.goal_category == "retrieval":
                if config['goal_in_hand'] == True:
                    self.object_in_hand = PlanParser.create_legal_object_name(config['id_goal_in_hand'])
                self.goal_object_id = PlanParser.create_legal_object_name(config['goal']['metadata']['target']['id'])
                for obj in config['objects']:
                    if obj['id'] == self.goal_object_id:
                        continue
                    if "openable" in obj and obj["openable"] == True:
                        receptacle_object_id = PlanParser.create_legal_object_name(obj['id'])
                        if "opened" in obj:
                            self.object_open_close_info[receptacle_object_id] = True
                        else:
                            self.object_open_close_info[receptacle_object_id] = False
                self.goal_predicate_list.append(
                    "(held {} {})".format(self.AGENT_NAME, self.goal_object_id)
                )
            elif self.goal_category == "transferral":
                self.transfer_object_id = PlanParser.create_legal_object_name(
                    config['goal']['metadata']['target_1']['id']
                )
                if config['goal_in_hand'] == True:
                    self.object_in_hand = PlanParser.create_legal_object_name(config['id_goal_in_hand'])
                self.target_object_id = PlanParser.create_legal_object_name(config['goal']['metadata']['target_2']['id'])
                for obj in config['objects']:
                    if obj['id'] == self.transfer_object_id:
                        continue
                    if "openable" in obj and obj["openable"] == True:
                        receptacle_object_id = PlanParser.create_legal_object_name(obj['id'])
                        if "opened" in obj:
                            self.object_open_close_info[receptacle_object_id] = True
                        else:
                            self.object_open_close_info[receptacle_object_id] = False

                if config['goal']['metadata']['relationship'][1] == "next to":
                    self.goal_predicate_list.append(
                        "(objectNextTo {} {})".format(self.transfer_object_id, self.target_object_id)
                    )
                elif config['goal']['metadata']['relationship'][1] == "on top of":
                    self.goal_predicate_list.append(
                        "(objectOnTopOf {} {})".format(self.transfer_object_id, self.target_object_id)
                    )






class GameKnowlege:
    def __init__(self):
        self.canNotPutin = set()
        self.objectNextTo = {}
        self.objectOnTopOf = {}
