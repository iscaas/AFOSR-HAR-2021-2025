#!/bin/bash
echo $config 

#module load Miniconda3          # load conda module
eval "$(conda shell.bash hook)" # Initialize the shell to use Conda
conda info --envs               # list all conda envs available
conda activate 'env_name'


# Following was used initially, but currently testing whether this can by bypassed through a wandb login on headnode
# Set your wandb username as ENTITY and go to https://wandb.ai/authorize for API_KEY
# WANDB_ENTITY=yourwandbusername
# WANDB_API_KEY='long string'

time python tools/test.py $config


