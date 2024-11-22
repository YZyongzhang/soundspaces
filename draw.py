from matplotlib import pyplot as plt
from utils.angles import *
import math
from habitat.utils.visualizations import maps
class Draw:
    def __init__(self):
        pass
    def display_obs(self,obs):
        from habitat_sim.utils.common import d3_40_colors_rgb
        rgb_obs1, rgb_obs2 = obs[0]["camera"], obs[1]["camera"]
        rgb_img1 = Image.fromarray(rgb_obs1, mode="RGBA")
        rgb_img2 = Image.fromarray(rgb_obs2, mode="RGBA")

        arr = [rgb_img1, rgb_img2]
        titles = ["rgb1", "rgb2"]

        plt.figure(figsize=(12, 8))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i + 1)
            ax.axis("off")
            ax.set_title(titles[i])
            plt.imshow(data)
        plt.show(block=False)

    def convert_points_to_topdown(self , pathfinder, points, meters_per_pixel=0.01):
        points_topdown = []
        bounds = pathfinder.get_bounds()
        for point in points:
            # convert 3D x,z to topdown x,y
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return points_topdown


    # display a topdown map with matplotlib
    def display_map(self , topdown_map, key_points=None):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        plt.imshow(topdown_map)
        # plot points on map
        if key_points is not None:
            for point in key_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
        plt.show(block=False)


    def get_td_map(self , pathfinder, meters_per_pixel=0.01, vis_points=None):

        height = pathfinder.get_bounds()[0][1]
        xy_vis_points = None

        if vis_points is not None:
            xy_vis_points = self.convert_points_to_topdown(
                pathfinder, vis_points, meters_per_pixel
            )

        hablab_topdown_map = maps.get_topdown_map(
            pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]

        return hablab_topdown_map, xy_vis_points

    def add_agent_pos_angle(self , pathfinder, top_down_graph, agent_pos, agent_q):
        agent_angle = quat_to_angle(agent_q)
        show_angle = math.atan2(agent_angle[0], agent_angle[2])
        grid_dimensions = (top_down_graph.shape[0], top_down_graph.shape[1])

        grid_pos = maps.to_grid(
            agent_pos[2],
            agent_pos[0],
            grid_dimensions,
            pathfinder=pathfinder,
        )

        maps.draw_agent(
            top_down_graph, grid_pos, show_angle, agent_radius_px=16
        )

        return top_down_graph
    def show_graph(self , env):
        source_pos = env.get_source_pos()
        agent_pos = env.get_agent_pos()
        vis_points = source_pos
        x, y = self.get_td_map(env._sim.pathfinder, vis_points=vis_points)
        agent_r = env.get_agent_rotation()

        for agent_id in range(2):
            x = self.add_agent_pos_angle(env._sim.pathfinder, x, agent_pos[agent_id], agent_r[agent_id])

        self.display_map(x,y)
    def draw(self, sim , path):
        meters_per_pixel = 0.025
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        height = scene_bb.y().min
        top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                )
        recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
        top_down_map = recolor_map[top_down_map]
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        trajectory = [
                    maps.to_grid(
                        path_point_[2],
                        path_point_[0],
                        grid_dimensions,
                        pathfinder=sim.pathfinder,
                    )
                    for path_point_ in path
                ]
        grid_tangent = mn.Vector2(
                    trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                )
        path_initial_tangent = grid_tangent / grid_tangent.length()
        initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
        maps.draw_path(top_down_map, trajectory)
        maps.draw_agent(
        top_down_map, trajectory[0], initial_angle, agent_radius_px=8
        )
        return top_down_map