import os, argparse, random

parser = argparse.ArgumentParser(description='')
# EXPERIMENT SETTINGS
parser.add_argument('--target_cluster', type=str, default="beluga", help='Target cluster name')
parser.add_argument('--store_folder', type=str, default="generated", help='Folder to store the generated script')
parser.add_argument('--docker', type=int, default=1, help='Use Singularity container (1) or not (0)')
parser.add_argument('--folder_venv', type=str, default='~/ENV', help='the folder holding the virtual environment')
parser.add_argument('--implementation', type=str, default='pytorch', help='pytorch or tensorflow')

parser.add_argument('--runtimes', type=int, default=1, help='Runtimes.')
parser.add_argument('--network', type=str, default='gcn_onepath', help='Network type (snowball, linear_snowball, linear_tanh_snowball, truncated_krylov)')
parser.add_argument('--public', type=int, default=1, help='Runtimes.')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset (Cora, Citeseer, Pubmed)')
# HYPERPARAMETERS
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.007, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=6e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.3, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()

if args.implementation == 'pytorch':
    folder_exec = '~/scratch/GAT_twopaths'
    location_image = '~/scratch/pytorch_20.01-py3.sif'
elif args.implementation == 'tensorflow':
    folder_exec = '~/scratch/Stronger-GCN-TF2'
    args.docker = 0

if args.target_cluster == 'helios':
    args.runtimes = 10

suffix = ""
args_dict = args.__dict__
for arg in args_dict:
    if arg == "target_cluster":
        target_cluster = args_dict[arg]
    elif arg == "store_folder":
        store_folder = args_dict[arg]
        try:
            os.mkdir(store_folder)
        except OSError:
            pass
    elif arg == "docker" or arg == "implementation" or arg == "folder_venv":
        pass
    else:
        suffix = suffix + "--%s %s " % (arg, str(args_dict[arg]))
identifier = random.randint(0, 1e6)
suffix = suffix + "--identifier %d" % identifier
filename = target_cluster + "_" + str(identifier) + ".sh"
filepath = store_folder + "/" + filename
script = open(filepath, 'w')
script.write("#!/bin/bash\n")
if target_cluster == "helios":
    script.write("#PBS -N MyJob" + "\n")
    script.write("#PBS -A jvb-000-ag" + "\n")
    script.write("#PBS -l walltime=5:59:0" + "\n")
    script.write("#PBS -l nodes=1:gpus=1" + "\n")
    script.write("\n")
elif target_cluster == "beluga":
    script.write("#SBATCH --account=rpp-bengioy" + "\n")
    script.write("#SBATCH --cpus-per-task=2" + "\n")
    script.write("#SBATCH --gres=gpu:1" + "\n")
    script.write("#SBATCH --mem=8G" + "\n")
    script.write("#SBATCH --time=2:59:59" + "\n")
    script.write("\n")
    script.write("module load cuda/10.0" + "\n")
elif target_cluster == "graham":
    script.write("#SBATCH --account=def-bengioy" + "\n")
    script.write("#SBATCH --cpus-per-task=1" + "\n")
    script.write("#SBATCH --gres=gpu:1" + "\n")
    script.write("#SBATCH --mem=8G" + "\n")
    script.write("#SBATCH --time=2:59:59" + "\n")
    script.write("\n")
    script.write("module load cuda/10.0" + "\n")
script.write("cd %s" % folder_exec + "\n")
if args.docker:
    script.write("module load singularity/3.5" + "\n")
    script.write("singularity exec --nv %s python train.py %s" % (location_image, suffix) + "\n")
else:
    script.write("module load python/3.7 scipy-stack" + "\n")
    script.write("source %s/bin/activate" % args.folder_venv + "\n")
    script.write("python train.py %s" % suffix + "\n")
script.close()
print(filename)