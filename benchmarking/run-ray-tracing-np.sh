#!/bin/bash
#SBATCH --job-name=ray_tracing
#SBATCH --time=2-00:00:00              # Adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # Single task for Python script
#SBATCH --cpus-per-task=1            # Cores for parallel CPU work if needed
#SBATCH --mem=8G                    # Adjust based on scene complexity

#SBATCH --partition=kingspeak-gpu
##SBATCH --qos=kingspeak-gpu
##SBATCH --account=kingspeak-gpu
##SBATCH --gres=gpu:1          # Any GPU on that partition


# Option 2: A40 on notchpeak
##SBATCH --partition=notchpeak-gpu
##SBATCH --account=notchpeak-gpu
##SBATCH --gres=gpu:a40:1

# Option 3: RTX 3090 on notchpeak (good fallback, likely shorter queue)
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:3090:1

# Option 4: Guest access to owner nodes (may be preempted)
##SBATCH --partition=notchpeak-gpu-guest
##SBATCH --account=owner-gpu-guest
##SBATCH --gres=gpu:rtx6000:1

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=thomas.dewitt@utah.edu

#SBATCH --error=slurm-%j.err

# Load miniconda module
module use ~/mymodules
module load miniconda3

# Hardcode conda base path from your env list
source /uufs/chpc.utah.edu/common/home/u1020524/software/pkg/miniconda3/etc/profile.d/conda.sh
conda activate base




# Debug: show what we're using
echo "Python in job: $(which python)"
echo "Behold in job: $(which behold)"


# Set working directory
cd /uufs/chpc.utah.edu/common/home/u1020524/cloudyview/benchmarking/

# Verify GPU allocation
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOBID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo ""

# Check GPU detection
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi -L
    echo ""
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo ""
else
    echo "ERROR: nvidia-smi not found!"
    exit 1
fi



# Run your Python ray tracing script
python benchmark.py

echo "Job completed on $(date)"
