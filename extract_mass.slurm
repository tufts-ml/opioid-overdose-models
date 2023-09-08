#!/usr/bin/env bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 00-04:00          # Runtime in D-HH:MM
#SBATCH -p interactive           # Partition to submit to
#SBATCH --mem-per-cpu 80000  # Memory (in MB) per cpu
#SBATCH -o /cluster/tufts/hugheslab/kheuto01/slurmlog/out/log_%j.out       # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/kheuto01/slurmlog/err/log_%j.err       # Write stderr to file named log_JOBIDNUM.err in log dir


python /cluster/home/kheuto01/code/opioid-overdose-models/massachusetts/cleaning_massachusetts/extract_dataset.py --output_dir "/cluster/tufts/hugheslab/datasets/NSF_OD/results_202308_pipeline" --do_final_quarterly
