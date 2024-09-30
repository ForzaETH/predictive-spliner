#!/usr/bin/env python3
import rospy
from dynamic_reconfigure.server import Server
from predictive_spliner.cfg import dyn_sqp_tunerConfig

def callback(config, level):
    config.evasion_dist = round(config.evasion_dist, 2)
    config.obs_traj_tresh = round(config.obs_traj_tresh,2)
    config.spline_bound_mindist = round(config.spline_bound_mindist, 3)
    config.lookahead_dist = round(config.lookahead_dist, 2)
    config.avoidance_resolution = round(config.avoidance_resolution)
    config.back_to_raceline_before = round(config.back_to_raceline_before, 2)
    config.back_to_raceline_after = round(config.back_to_raceline_after, 2)

    config.avoid_static_obs = config.avoid_static_obs
    return config

if __name__ == "__main__":
    rospy.init_node("dynamic_sqp_tuner_node", anonymous=False)
    print('[Planner] Dynamic SQP Server Launched...')
    srv = Server(dyn_sqp_tunerConfig, callback)
    rospy.spin()