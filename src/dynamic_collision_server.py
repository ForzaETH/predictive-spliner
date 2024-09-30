#!/usr/bin/env python3
import rospy
from dynamic_reconfigure.server import Server
from predictive_spliner.cfg import dyn_collision_tunerConfig

def callback(config, level):
    config.n_time_steps = round(config.n_time_steps)
    config.dt= config.dt
    config.save_distance_front = round(config.save_distance_front, 2)
    config.save_distance_back = round(config.save_distance_back, 2)
    config.max_v = round(config.max_v, 2)
    config.min_v = round(config.min_v, 2)
    config.max_a = round(config.max_a, 2)
    config.min_a = round(config.min_a, 2) 
    config.max_expire_counter = round(config.max_expire_counter)  
    config.update_waypoints = config.update_waypoints
    config.speed_offset = round(config.speed_offset, 3)
    return config

if __name__ == "__main__":
    rospy.init_node("dynamic_collision_tuner_node", anonymous=False)
    print('[Planner] Dynamic Collision Server Launched...')
    srv = Server(dyn_collision_tunerConfig, callback)
    rospy.spin()
