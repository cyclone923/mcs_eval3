
import matplotlib.pyplot as plt

class GeometryPlot:

    def __init__(self):
        plt.figure(figsize=(2, 2))
        self.ax = plt.axes()

    def plot_view(self, poly, color='-b'):
        if poly.geom_type == 'MultiPolygon':
            for i in range(len(poly)):
                pts = poly[i].exterior.coords
                self.ax.fill([p[0] for p in pts], [p[1] for p in pts], color)
        else:
            pts = poly.exterior.coords
            self.ax.fill([p[0] for p in pts], [p[1] for p in pts], color)

    def plot_obstacles(self, obstacles):
        for obstacle in obstacles:
            obstacle.plot(self.ax, "green")

    def plot_agent(self, x, z, radius):
        circle = plt.Circle((x, z), radius=radius, color='r')
        self.ax.add_artist(circle)

    def plot_nav_info(self, nav):
        x, z, _ = nav.goal
        self.ax.plot(x, z, 'x', c='k', markersize=3)
        for x, z in zip(nav.path_x, nav.path_z):
            self.ax.plot(x, z, "x", c='k', markersize=0.3)


    def show_plot(self, agent):
        self.ax.clear()
        self.ax.axis('equal')
        self.plot_agent(agent.agent_x, agent.agent_z, agent.agent_radius)
        self.plot_view(agent.explore_strategy.world_poly, color='-y')
        self.plot_view(agent.explore_strategy.current_view)
        self.plot_obstacles(agent.scene_obstacles.values())
        if agent.navigator.goal:
            self.plot_nav_info(agent.navigator)
        plt.pause(0.00001)