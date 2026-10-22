# A moment in the pick cycle. header.stamp = when the robot reached it.
string CYCLE_START="cycle_start"
string PRE_PICK="pre_pick"
string GRASP_CLOSED="grasp_closed"
string PLACE_DONE="place_done"

std_msgs/Header header
string event
int32 trial_id
string object_cls       # what this pick is for (from the run sheet or the program slot)
string pick_point       # name in config pick_points, e.g. slot_1
string target_bin       # name in config bins; empty = object's default bin
string source           # joint_states | twin | manual | guide
# Guide only, on PRE_PICK: the object pose it is about to grab (table frame). The pre-pick
# check then compares against this planned grasp instead of the program's fixed pick point.
bool has_target
float64 target_x_mm
float64 target_y_mm
float64 target_yaw_deg
