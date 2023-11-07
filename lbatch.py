import os.path as path
import sys
import argparse
import subprocess

def writesh(pyFile: str, cluster: str="MAHTI", np: int=0, ntsk: int=0, nnd: int=0, ppnd: int=0, tskpnd: int=0, pptsk: int=1,\
    mpp: int=0, e_notif: str="", account: str="khonkala", partition: str="", timelimit: str="", mod_gpaw: str="", singularity: bool=False) -> str:
    """Creates a Slurm script file that runs the specified script with gpaw-python
    pyFile: Python script to run. Required
    cluster: Name of the computing cluster [Mahti, Puhti, Oberon, or Puck]
    nnd: Number of nodes to split the job onto [SBATCH: --nodes] Set to 0 to calculate automatically
    mpp: Memory per processor [] set to 0 for default/to exclude
    e_notif: Give your email address, if you want a notification when the calcuations stop"""
    
    pyFile = path.abspath(pyFile)
    namn = pyFile
    batch = ""
    csc = False

    if mod_gpaw != "":
        if mod_gpaw[-1] != "/": mod_gpaw += "/"
    
    if namn.__contains__("."):
        namn = namn[:namn.rfind(".")]
    
    if namn.__contains__("/"):
        Onamn = namn[namn.rfind("/")+1:]
        OpyFile = pyFile[pyFile.rfind("/")+1:]
    else: OpyFile = pyFile

    shFile = path.abspath(f"{namn}.sh")


    if cluster.upper() == "MAHTI":
        if ppnd == 0: ppnd = 128
        if mod_gpaw != "":
            # modle = f"module load gpaw/20.1.0-omp\nGPAW={mod_gpaw}lib/python3.8/site-packages\nexport PYTHONPATH=$GPAW\n\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\nexport OMP_PLACES=cores"
            modle = f"module load gpaw/20.10.0-omp\n\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\nexport OMP_PLACES=cores"
        else:
            modle = "module load gpaw/20.10.0-omp\n\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\nexport OMP_PLACES=cores"
        if partition == "": partition = "medium"
        if timelimit == "": timelimit = "36:00:00"
        csc = True
    elif cluster.upper() == "PUHTI":
        if ppnd == 0: ppnd = 40
        modle = "module load mpich/3.3.1\nmodule load python-env\nmodule load gpaw"
        if mpp == 0: mpp = 4
        if partition == "": partition = "large"
        if timelimit == "": timelimit = "72:00:00"
        csc = True
    elif cluster.upper() == "OBERON":
        if ppnd == 0: ppnd = 40
        modle = "module load gpaw/1.5.1-gcc"
    elif cluster.upper() == "PUCK":
        if ppnd == 0: ppnd = 24
        modle = "module load puck_gpaw/1.5.1-ase3.17.0-gcc"


    if nnd == 0: # If number of nodes is not specified
        nnd = 1 # Default to one
        if ppnd != 0: # If number of processors per node is specified (should always be): determine how many nodes would be needed
            while nnd*ppnd < np: 
                nnd = nnd + 1
    if np == 0: np = ppnd*nnd # If number of processors is not set, determine based on the number of nodes
    
    if ntsk == 0: ntsk = np # If number of tasks is not specified, use number of processors
    if tskpnd == 0: tskpnd = ppnd# If number of tasks per node is not specified, use number of processors per node

    if pptsk > 1:
        tskpnd = ppnd // pptsk
        ntsk = tskpnd * nnd
        np = ppnd * nnd


    batch = f"#!/bin/bash -l\n#SBATCH -J {Onamn}\n#SBATCH -o {Onamn}.out\n#SBATCH -e {Onamn}.err\n"
    batch += f"#SBATCH --ntasks={ntsk}\n"
    batch += f"#SBATCH --ntasks-per-node={tskpnd}\n#SBATCH --nodes={nnd}\n"
    batch += f"#SBATCH --cpus-per-task={pptsk}\n"

    if mpp != 0: batch += f"#SBATCH --mem-per-cpu={mpp}GB\n"

    if partition != "": batch += f"#SBATCH --partition={partition}\n"

    if timelimit != "": batch += f"#SBATCH -t {timelimit}\n"

    if csc: 
        batch += f"#SBATCH --account={account}\n"
        if singularity: modle += "\nmodule load python-singularity/3.8.2"

    if e_notif != "": batch += f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={e_notif}\n"

    batch += f"\n{modle}\n\n"

    if csc:
        if mod_gpaw != "":
            batch += f"srun {mod_gpaw}bin/gpaw-python {OpyFile}"
        else:
            batch += f"srun gpaw-python {OpyFile}"
    else: batch += f"mpirun -np {np} gpaw-python {OpyFile}"

    with open(shFile,"w") as w:
        w.write(f"{batch}\n")
    
    if shFile.__contains__("/"):
        shFile = shFile[shFile.rfind("/")+1:]
            
    return shFile


def writepy(name: str="job", tem: str="~/pytem.py") -> str:
    inExt = "traj"

    t_name = path.abspath(path.expanduser(name))
    if t_name.__contains__("@"): t_name = t_name[:t_name.find("@")]

    if path.exists(t_name):
        if name.__contains__("."):
            name, inExt = name.rsplit(".", maxsplit=1)
        
        if inExt == "py": raise TypeError
        pyFile = name + ".py"
        
        pytxt = ""
        tem = path.abspath(path.expanduser(tem))
        try:
            with open(tem, "r") as puh:
                for l in puh:
                    if l.__contains__("?name?"): l = l.replace("?name?", str(name))
                    if l.__contains__("?inExt?"): l = l.replace("?inExt?", str(inExt))
                    pytxt = pytxt + l
        except FileNotFoundError:
            pytxt = f"The template file was not found in {tem}"

        with open(pyFile,"w") as w:
            w.write(f"{pytxt}\n")
        
        return pyFile
    else:
        return "_na"


if __name__ == "__main__":
    cluster = "Mahti"
    account = "khonkala"
    lbfolder = path.dirname(path.realpath(__file__))
    pytem = path.abspath(lbfolder + "/pytem.py")
    nebtem = path.abspath(lbfolder + "/nebtem.py")
    
    arpar = argparse.ArgumentParser(description="Creates a SLURM script to run a specified file")

    arpar.add_argument("file", type=str, help="The file containing the atomic coordinates")
    arpar.add_argument("-v", action="store_true", help="Verbose. By default lbatch returns only the name of the SLURM script file")
    arpar.add_argument("-c", metavar="CLUSTER", type=str, default=cluster, 
                        help=f"The name of the cluster the job will be run on. This allows the code to generate suitable default values. Defaults to \"{cluster}\"")
    arpar.add_argument("-a", metavar="ACCOUNT", type=str, default=account, help=f"The account used for computer time when running on CSC supercomputers. Defaults to \"{account}\"")
    arpar.add_argument("-g", action="store_true", help="Uses the modified installation of GPAW available on the khonkala project")
    arpar.add_argument("-p", type=int, default=0, help="Number of processors")
    arpar.add_argument("-n", type=int, default=0, help="Number of nodes")
    arpar.add_argument("-m", type=float, default=0., help="Memory per processor (GB)")
    arpar.add_argument("-t", type=int, default=0, help="Time limit in hours (int). There's no sanity checking here so check the limit for each cluster")
    arpar.add_argument("-s", action="store_true", help="Automatically submits the job to the SLURM system")
    arpar.add_argument("--sin", action="store_true", help="Loads the 'Python singularity' containing scipy, pandas, matplotlib etc. DON'T USE THIS")
    arpar.add_argument("--ppnd", type=int, default=0, help="Processors per node")
    arpar.add_argument("--part", metavar="PARTITION", type=str, default="", help="The SLURM partition to use. Defaults to \"medium\" on Mahti and \"large\" on Puhti")
    arpar.add_argument("--email", type=str, default="", help="You can give an email address to recieve a notification of events during the job's execution")
    arpar.add_argument("--sh", action="store_true", help="Generates ONLY the SLURM script file. \"file\" should then contain the script to be run, eg. a python file")
    arpar.add_argument("--neb", action="store_true", help="Generates a file to run a Nudged Elastic Band (NEB) calculation")

    args = arpar.parse_args()

    fil = args.file

    verbose = args.v

    timelimit = args.t
    if timelimit != 0:
        timelimit = f"{str(timelimit).rjust(2,'0')}:00:00"
    else:
        timelimit = ""
    
    if args.g: mod_gpaw = "/projappl/khonkala/21.1.0-gcc-openblas/"
    else: mod_gpaw = ""
        
    if args.neb: 
        pytem = nebtem
    
    lazy = not args.sh


    if lazy:
        if verbose: print("Python file:")
        _fil = writepy(name=fil, tem=pytem)
        if _fil == "_na":
            print(f"No file \"{fil}\" found")
        elif verbose:
            print(_fil)
            print("It is recommended that you check the parameters before submitting")
        fil = _fil


    fil = path.abspath(fil)
    
    if path.exists(fil):
        sph = writesh(pyFile=fil, cluster=args.c, np=args.p, nnd=args.n, ppnd=args.ppnd, mpp=args.m,
                      e_notif=args.email, account=args.a, partition=args.part, mod_gpaw=mod_gpaw,
                      timelimit=timelimit, singularity=args.sin)
        if verbose: print("SLURM batch file:")

        if args.s:
            sproc = subprocess.run(f"sbatch {sph}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            sprout = sproc.stdout.decode()
            print(sprout[:sprout.rfind("\n")])
        else:
            print(sph)
    
    elif fil == "_na": print("No scripts were generated")
    else: print(f"No file \"{fil}\" found")
    
    