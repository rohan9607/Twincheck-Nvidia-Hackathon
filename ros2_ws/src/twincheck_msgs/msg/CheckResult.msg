std_msgs/Header header
int32 trial_id
string check_point
bool passed
float64 position_error_mm   # NaN when not measured
float64 yaw_error_deg       # NaN when not measured
string reason
string action               # none | alert | retry | stop
bool urgent
string expected_img
string observed_img
float64 sync_gap_ms
string expected_source      # config | twin
