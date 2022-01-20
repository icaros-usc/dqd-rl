# Config for 100 cores with high memory. Typically used when there is a noise
# table on each worker (since the table takes up 1GB).
HPC_SLURM_ACCOUNT=nikolaid_548
HPC_SLURM_TIME=24:00:00
HPC_SLURM_NUM_NODES=10
HPC_SLURM_CPUS_PER_NODE=10
HPC_SLURM_MEM_PER_CPU=1700MB
