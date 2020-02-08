### Time window 
TIME_WINDOW_SIZE = 20 # Seems ok, at 20hz the lstm receive 1 sec of data to anticipate a risk
OVERLAP_RATIO = 0.5     # ratio in ]0,1[  ratio=0.2 means the step will be 20% of the TW_SIZE
STEP = int(TIME_WINDOW_SIZE/OVERLAP_RATIO)
### IA
# modes=['cartesian_coord','distance_mat'] # we could had anything, using speed or whatever
# MODE=modes[0]
BATCH_SIZE = 128 # As big as possible without crashing the reaining process
EPOCHS = 5       # IDK
TEST_SIZE = 0.5  # IDK