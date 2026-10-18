# What should be true at a check point. Published by the expectation node
# (from config) and, when running, by the Isaac Sim shadow twin (its prediction).
std_msgs/Header header
int32 trial_id
string check_point          # pre_pick | post_grasp | post_place
bool object_expected        # false if the twin predicts the grasp will fail
DetectedObject expected_object
uint8 expected_grasp        # GraspState constant
string target_bin
string source               # config | twin
string expected_img         # path to a rendered image of the expected scene, if any
