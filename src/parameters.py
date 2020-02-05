### Time window 
TIME_WINDOW_SIZE = 20
OVERLAP_RATIO = 2
STEP = int(TIME_WINDOW_SIZE/OVERLAP_RATIO)

### IA
modes=['cartesian_coord','distance_mat'] # we could had anything, using speed or whatever
MODE=modes[0]
LSTM_SIZE = 512
BATCH_SIZE = 64
EPOCHS = 5
TEST_SIZE = 0.5