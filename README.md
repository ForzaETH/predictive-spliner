# Predictive Spliner
The predictive_spliner package together with the opponent_trajectory package in the perception folder contains most of the files to run *predictive spliner*.
To run *predictive spliner* during head-to-head racing, run:
```
roslaunch stack_master headtohead.launch LU_table:=SIM_linear overtake_mode:=predictive_spliner solver:=sqp od_mode:=banana GP_trajectory:=gp_predictive
```
For *predictive spliner*, there are two different solver options: `sqp` and `spliner`. `sqp` utilizes dynamic programming techniques to generate an overtaking trajectory, while `spliner` works similarly to *spliner*, only with an enlarged region of collision. The default and recommended solver setting is `sqp`.

In simulation, `GP_trajectory` can be set to `false`. This will use the ground truth of the opponents trajectory published by the *obstacle publisher node*.
In head_to_head mode set `GP_trajectory` to `gp_predictive` to utilize the predicted opponent raceline trajectory.
To utilize websocket functionalities, launch car to car sync to stran over the data from the opponent to the ego car and set `GP_trajectory` to `websocket`.

### RQT Parameters
The default settings work fine for a somewhat consistent opponent. If the opponent drives more unpredictably, the `save_distance_front`, `save_distance_back`, and `evasion_dist` should be increased. If the computational time for the waypoint generation is too slow, the `avoidance resolution` can be decreased. It is recommended to choose a value higher or equal to 18. By default, `avoid static` obstacles is deactivated. Activate it if there is a chance of static obstacles appearing on the track.

### Spliner Remark
The `spline_ttl` parameter has been changed to 0.2; the default value for "normal" *spliner* is 2. If you use "normal" spliner, you need to change this value back to 2, so that *spliner* can function properly.


