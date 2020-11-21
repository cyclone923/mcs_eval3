class TrophyGoal:

    def __init__(self):
        self.position = None
        self.goal_found = False

    def reset(self):
        self.position = None
        self.goal_found = False

    def __str__(self):
        s = ""
        if self.goal_found:
            s = f"({self.position[0]:.3f}, {self.position[1]:.3f})"
        return s

    def update_trophy_goal_oracle(self, step_output):
        for obj in step_output.object_list:
            if obj.uuid == "trophy" and obj.visible:
                self.position = (obj.position['x'], obj.position['z'], None)
                self.goal_found = True
                break

    def update_trophy_goal_level1(self, step_output):
        return

