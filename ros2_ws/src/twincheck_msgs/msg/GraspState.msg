# What the wrist camera sees between the fingers. header.stamp = wrist capture time.
uint8 HOLDING=0
uint8 EMPTY=1
uint8 MISALIGNED=2
uint8 UNKNOWN=3

std_msgs/Header header
uint8 state
float64 confidence
string method      # reference_diff | classifier | cosmos
string detail      # free text, e.g. the VLM's one-line answer
