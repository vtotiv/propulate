#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --output=surrogate_%j.out
#SBATCH --error=surrogate_%j.err
#SBATCH --job-name=surrogate

# load module
module purge
module load devel/python/3.10.5_gnu_12.1
module load compiler/gnu/12.1
module load mpi/openmpi/4.1
module load devel/cuda/12.2

# get relevant branch
git clone -b bwunicluster https://github.com/vtotiv/propulate.git
cd propulate

python -m venv ./propulateenv
source ./propulateenv/bin/activate
pip install --upgrade pip

# DO NOT REMOVE - change python version here in case of module upgrade
export MPICC=$(which mpicc)
export PYTHONPATH=/pfs/data5/home/kit/scc/$(whoami)/.local/lib/python3.10/site-packages:$PYTHONPATH

# install all dependencies
pip install -r requirements.txt
pip install -r tutorials/surrogate/requirements.txt
pip install perun

pip install -e .

# run - change to number of GPUs
mpirun -np 2 perun monitor tutorials/surrogate/mnist_example.py
