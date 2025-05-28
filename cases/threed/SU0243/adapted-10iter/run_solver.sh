#!/bin/bash

#name of your job 
#SBATCH --job-name=svFlowSolver
#SBATCH --partition=amarsden

# Specify the name of the output file. The %j specifies the job ID
#SBATCH --output=svFlowSolver.o%j

# Specify the name of the error file. The %j specifies the job ID 
#SBATCH --error=svFlowSolver.e%j

# The walltime you require for your job 
#SBATCH --time=16:00:00

# Job priority. Leave as normal for now 
#SBATCH --qos=normal

# Number of nodes are you requesting for your job. You can have 24 processors per node 
#SBATCH --nodes=4 

# Amount of memory you require per node. The default is 4000 MB per node 
#SBATCH --mem=16G

# Number of processors per node 
#SBATCH --ntasks-per-node=24 

# Send an email to this address when your job starts and finishes 
#SBATCH --mail-user=ndorn@stanford.edu 
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end 
module --force purge

ml devel
ml math
ml openmpi
ml openblas
ml boost
ml system
ml x11
ml mesa
ml qt
ml gcc/14.2.0
ml cmake

srun /home/users/ndorn/svMP-procfix/svMP-build/svMultiPhysics-build/bin/svmultiphysics svFSIplus.xml
