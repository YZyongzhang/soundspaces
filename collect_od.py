from config import config
# import attr
from env.v0d0 import Env
# import torch
import numpy as np
import habitat_sim
from habitat.utils.visualizations import maps
import random
from PID import NextPoint
from draw import Draw
random.seed(43)

env = Env(config)
env.reset()
sim = env._sim

aim_sound_point = env.get_source_pos()[0]
agent_state = env.get_agent_pos()

drawer = Draw()
path_0 = habitat_sim.ShortestPath()
path_0.requested_start = agent_state[0]
path_0.requested_end = aim_sound_point
sim.pathfinder.find_path(path_0)
pic = drawer.draw(sim,path_0.points)
drawer.display_map(pic)

print('waypoints are as below')

for  i  in range(5):

    print("begin")
    end = sim.pathfinder.get_random_navigable_point()
    drawer = Draw()
    path_0 = habitat_sim.ShortestPath()
    path_0.requested_start = agent_state[0]
    path_0.requested_end = end
    sim.pathfinder.find_path(path_0)
    pic = drawer.draw(sim,path_0.points)
    drawer.display_map(pic)
    print("above is the target")

    next_position = NextPoint()
    # drawer = Draw()
    [x,z,y]= agent_state[0]
    path = list()
    path_ = habitat_sim.ShortestPath()
    for i in range(5):
        path_.requested_start = [x,z,y]
        x,y = next_position.get_next_position(path_.requested_start , end)
        path_.requested_end = [x,z,y]
        sim.pathfinder.find_path(path_)
        pic = drawer.draw(sim,path_.points)
        drawer.display_map(pic)
    print("end!")