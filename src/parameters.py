### Time window 
TIME_WINDOW_SIZE = 20 # Seems ok, at 20hz the lstm receive 1 sec of data to anticipate a risk
OVERLAP_RATIO = 2     # IDK
STEP = int(TIME_WINDOW_SIZE/OVERLAP_RATIO)

### IA
modes=['cartesian_coord','distance_mat'] # we could had anything, using speed or whatever
MODE=modes[0]

LSTM_SIZE = 256  # easy formula can be, Nhidden_node = (2/3)*(Ninput + Noutput) = (2/3)*(16*TW + Nb_classes) = (2/3)*(360 + 4)
BATCH_SIZE = 128 # As big as possible without crashing the reaining process
EPOCHS = 5       # IDK
TEST_SIZE = 0.5  # IDK