#!/usr/bin/env bash

# copy model.py to AWS server
scp ./model.py carnd@$1:~/CarND-Behavioral-Cloning-P3

# scp data/02_track1_forward.zip carnd@$1:~/CarND-Behavioral-Cloning-P3/data/
# scp data/03_track1_backward.zip carnd@$1:~/CarND-Behavioral-Cloning-P3/data/
# scp data/04_track2_forward.zip carnd@$1:~/CarND-Behavioral-Cloning-P3/data/
# scp data/05_track2_forward.zip carnd@$1:~/CarND-Behavioral-Cloning-P3/data/



# copy models directory to local machine
scp -r carnd@$1:~/CarND-Behavioral-Cloning-P3/models/ .
