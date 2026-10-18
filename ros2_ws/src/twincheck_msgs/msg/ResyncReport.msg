# After each pick: how far twin objects had to move to match the cameras.
std_msgs/Header header
int32 trial_id
string[] twin_ids
string[] classes
float64[] shift_mm          # NaN for newly added objects
string[] missing_twin_ids
