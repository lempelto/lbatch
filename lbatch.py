import os.path as path
import subprocess


vasp_fast = {
    "GGA": "PE",
    "ALGO": "Normal",
    "ENCUT": 400.,
    "PREC": "Accurate",
    "LASPH": True,
    "LREAL": "Auto",
    "IVDW": 12,
    "EDIFF": 1.0e-05,
    "ISMEAR": 1,
    "SIGMA": 0.05,
    "ISPIN": 1,
    "NELMIN": 8,
    "NELM": 150,
    "NELMDL": -8,
    "ISYM": 0,
    "IBRION": 2,
    "EDIFFG": -1.0e-02,
    "NSW": 500,
    "LDAU": True,
    "LDAUTYPE": 2,
    "LDAUL": "2 -1 2 -1 -1",
    "LDAUU": "6.20 0.00 4.40 0.00 0.00",
    "LDAUJ": "0.00 0.00 0.00 0.00 0.00",
    "LMAXMIX": 6,
    # "KPAR": 4,
    "LORBIT": 11,
    "LCHARG": True,
    "LWAVEH5": True
}

def write_sh(pyFile: str="", run_cmd: str="", cluster: str="", dft: str="", np: int=0, ntsk: int=0, nnd: int=0, ppnd: int=0, tskpnd: int=0, pptsk: int=1,\
    mpp: int=0, e_notif: str="", account: str="", partition: str="", timelimit: str="") -> str:
    """Creates a Slurm script file that runs the specified script with gpaw-python
    pyFile: Python script to run. Required
    cluster: Name of the computing cluster [Mahti, Puhti, Oberon, or Puck]
    nnd: Number of nodes to split the job onto [SBATCH: --nodes] Set to 0 to calculate automatically
    mpp: Memory per processor [] set to 0 for default/to exclude
    e_notif: Give your email address, if you want a notification when the calcuations stop"""

    _gpaw = dft == "GPAW"
    _vasp = dft == "VASP"
    
    pyFile = path.abspath(pyFile)
    namn = pyFile
    batch = ""
    csc = False
    
    if namn.__contains__("."):
        namn = namn[:namn.rfind(".")]
    
    if namn.__contains__("/"):
        Onamn = namn[namn.rfind("/")+1:]
        OpyFile = pyFile[pyFile.rfind("/")+1:]
    else: OpyFile = pyFile

    shFile = path.abspath(f"{namn}.sh")


    if cluster.upper() == "MAHTI":
        if ppnd == 0: ppnd = 128
        if _gpaw:
            modle = "module load gpaw/20.10.0-omp\n\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\nexport OMP_PLACES=cores"
        elif _vasp:
            modle = "module load gpaw\nmodule load vasp/6.4.3\n"
            modle += "export ASE_VASP_COMMAND=\"vasp_std\"\n"
            modle += "export VASP_PP_PATH=\"/scratch/project_2012891/\""
        if partition == "": partition = "medium"
        if timelimit == "": timelimit = "36:00:00"
        csc = True
    elif cluster.upper() == "PUHTI":
        if ppnd == 0: ppnd = 40
        if _gpaw:
            modle = "module load mpich/3.3.1\nmodule load python-env\nmodule load gpaw"
        elif _vasp:
            modle = "module load gpaw\nmodule load vasp/6.4.3\n"
            modle += "export ASE_VASP_COMMAND=\"vasp_std\"\n"
            modle += "export VASP_PP_PATH=\"/scratch/project_2012891/\""
        if mpp == 0: mpp = 8
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

    if e_notif != "": batch += f"#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user={e_notif}\n"

    batch += f"\n{modle}\n\n"

    if csc:
        if _gpaw:
            batch += f"srun gpaw-python {OpyFile}"
        elif _vasp:
            batch += f"srun python3 {OpyFile}"
    else:
        batch += f"mpirun -np {np} gpaw-python {OpyFile}"

    with open(shFile,"w") as w:
        w.write(f"{batch}\n")
    
    if shFile.__contains__("/"):
        shFile = shFile[shFile.rfind("/")+1:]
            
    return shFile


def write_py(inFile: str="", dft: str="", tem: str="~/pytem.py") -> str:
    """Writes Python input script to run a job with ASE"""
    inExt = ""

    t_name = path.abspath(path.expanduser(inFile))
    if t_name.__contains__("@"): t_name = t_name[:t_name.find("@")]

    if path.exists(t_name):
        if inFile.__contains__("."):
            name, inExt = inFile.rsplit(".", maxsplit=1)
        elif inFile == "POSCAR":
            name = f"job{str(hash(t_name))}"
        else:
            name = inFile
        
        # If someone's sent a Python file as molecular coordinates,
        # let's stop so that we don't accidentally overwrite it
        if inExt == "py": raise TypeError
        
        pyFile = name + ".py"

        pytxt = ""
        tem = path.abspath(path.expanduser(tem))
        try:
            with open(tem, "r") as puh:
                for l in puh:
                    if l.__contains__("?name?"):
                        l = l.replace("?name?", str(name))
                    if l.__contains__("?inFile?"):
                        l = l.replace("?inFile?", str(inFile))
                    pytxt = pytxt + l
        except FileNotFoundError:
            pytxt = f"The template file was not found in {tem}"

        with open(pyFile,"w") as w:
            w.write(f"{pytxt}\n")
        
        return pyFile
    else:
        return "_na"


def write_vasp(inFile: str="") -> str:
    """Writes input files for a Vasp calculation"""
    _incar = "INCAR"
    return _incar



if __name__ == "__main__":
    import platform
    import argparse


    # Try to guess which supercomputer we're on.
    # This can be overwritten by command line arguments
    hostname = platform.node()
    if hostname.__contains__("mahti"):
        cluster = "Mahti"
    elif hostname.__contains__("puhti"):
        cluster = "Puhti"
    else:
        cluster = "Mahti"

    account = "project_2012891"
    dft = "VASP"
    lbfolder = path.dirname(path.realpath(__file__))
    gtem = path.abspath(lbfolder + "/gpaw_tem.py")
    vtem = path.abspath(lbfolder + "/vasp_tem.py")
    nebtem = path.abspath(lbfolder + "/nebtem.py")
    
    arpar = argparse.ArgumentParser(prog="LBatch", 
                                    description="Creates the necessary files \
                                        to run a DFT calculation through SLURM")

    arpar.add_argument("file", type=str, 
                       help="The file containing the atomic coordinates")
    arpar.add_argument("-v", action="store_true", 
                       help="Verbose. By default lbatch returns only the name \
                            of the SLURM script file")
    arpar.add_argument("-d", metavar="DFT CODE", type=str, default=dft,
                        help=f"Which DFT package to run the calculations with.\
                            Currently the options are GPAW and VASP. Defaults \
                            to \"{cluster}\"")
    arpar.add_argument("-c", metavar="CLUSTER", type=str, default=cluster, 
                        help=f"The name of the cluster the job will be run on.\
                            This allows the code to generate suitable default \
                            values. Defaults to \"{cluster}\"")
    arpar.add_argument("-a", metavar="ACCOUNT", type=str, default=account,
                        help=f"The account used for computer time when running\
                            on CSC supercomputers. Defaults to \"{account}\"")
    arpar.add_argument("-p", type=int, default=0,
                        help="Number of processors")
    arpar.add_argument("-n", type=int, default=0, 
                       help="Number of nodes")
    arpar.add_argument("-m", type=float, default=0., 
                       help="Memory per processor (GB)")
    arpar.add_argument("-t", type=int, default=0, 
                       help="Time limit in hours (int). There's no sanity \
                            checking here so check the limit for each cluster")
    arpar.add_argument("-s", action="store_true",
                       help="Automatically submit the job to the SLURM system")
    arpar.add_argument("--ase", action="store_true", 
                       help="Uses ASE the ASE/Python interface to the DFT \
                            calculator. Enabled automatically when DFT CODE = \
                            \"GPAW\"")
    arpar.add_argument("--email", type=str, default="", 
                       help="You can give an email address to recieve a \
                        notification of events during the job's execution")
    arpar.add_argument("--sin", action="store_true",
                       help="Loads the 'Python singularity' containing scipy, \
                            pandas, matplotlib etc.")
    arpar.add_argument("--ppnd", type=int, default=0, 
                       help="Processors per node")
    arpar.add_argument("--part", metavar="PARTITION", type=str, default="", 
                       help="The SLURM partition to use. Defaults to \
                            \"medium\" on Mahti and \"large\" on Puhti")
    arpar.add_argument("--sh", action="store_true", 
                       help="Generates ONLY the SLURM script file. \"file\" \
                            should then contain the script to be run, eg. a \
                            python file")
    arpar.add_argument("--neb", action="store_true", 
                       help="Generates a file to run a Nudged Elastic Band \
                            (NEB) calculation using ASE")

    args = arpar.parse_args()

    fil = args.file
    dft = args.d.upper()
    use_ase = args.ase
    verbose = args.v
    lazy = not args.sh

    timelimit = args.t
    if timelimit != 0:
        timelimit = f"{str(timelimit).rjust(2,'0')}:00:00"
    else:
        timelimit = ""
    

    # Should not include the srun / mpirun command yet
    run_cmd = ""

    if args.neb:
        pytem = nebtem
    elif dft == "GPAW":
        use_ase = True
        run_cmd = "gpaw-python "
        pytem = gtem
    elif dft == "VASP":
        if use_ase:
            run_cmd = "python3 "
            pytem = vtem
        else:
            run_cmd = "vasp_std"

    
    # Lazy mode writes the Python file for ASE on the user's behalf.
    # I'm lazy so it's set as default. Can be disabled by using --sh
    if lazy:
        if use_ase:
            if verbose: print("Python file:")
            _fil = write_py(inFile=fil, dft=dft, tem=pytem)
            if _fil == "_na":
                print(f"No file \"{fil}\" found")
            elif verbose:
                print(_fil)
                print("It is recommended that you check the parameters before \
                      submitting")
            run_cmd += fil
        elif dft == "VASP":
            _fil = write_vasp(inFile=fil)
        else:
            raise NotImplemented
    else:
        _fil = fil

    fil = path.abspath(_fil)


    if path.exists(fil):
        sph = write_sh(run_cmd=run_cmd, pyFile=fil, cluster=args.c, dft=dft,
                       e_notif=args.email, account=args.a, partition=args.part,
                       np=args.p, nnd=args.n, ppnd=args.ppnd, mpp=args.m,
                       timelimit=timelimit)
        if verbose: print("SLURM batch file:")
        if args.s:
            sproc = subprocess.run(f"sbatch {sph}",
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
            sprout = sproc.stdout.decode()
            print(sprout[:sprout.rfind("\n")])
        else:
            print(sph)
    elif fil == "_na":
        print("No scripts were generated")
    else:
        print(f"No file \"{fil}\" found")
    
    
