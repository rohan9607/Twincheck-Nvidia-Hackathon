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
