
import os.path as path
import subprocess
import ltool
from config import config

class sh_writer:
    def __init__(self, dft: str, cluster: str="Mahti"):
        self.dft = dft.upper()
        self.cluster = cluster.upper()
        self.csc = True
        self.ase = False

        self.slurm: str
        self.modules: str
        self.procedure: str
        self.ase_pyfile: str

    def check_parallelization(self, 
                            np: int=0,
                            ntsk: int=0,
                            nnd: int=0,
                            ppnd: int=128,
                            tskpnd: int=0,
                            pptsk: int=1,
                            mpp: int=0,
                            **_) -> dict:

        if pptsk > 1:
            # If number of processors per task is more than 1, readjust the
            # value for tasks per node. Determines the number of tasks and the
            # number of processors, if not given.
            tskpnd = ppnd // pptsk
            if nnd == 0:
                if ntsk == 0:
                    ntsk = tskpnd * nnd
                if np == 0:
                    np = ppnd * nnd

        if nnd == 0:
            nnd = 1
            if ppnd != 0: 
                # If number of cores per node is specified (should always be):
                # add nodes until the requested number of processors is reached
                while nnd * ppnd < np: 
                    nnd = nnd + 1
            # Warning: if the user specifies both the number of nodes and an
            # unusable amount of processors, this will not check it. But let's
            # just call that user error.

        if np == 0:
            if ppnd != 0: 
                # If number of processors is not set, determine it based on the
                # number of nodes
                np = ppnd * nnd
        
        if ntsk == 0:
            # If number of tasks is not specified, use number of processors
            ntsk = np
            
        if tskpnd == 0:
            # If number of tasks per node is not specified, use number of
            # processors per node
            tskpnd = ppnd

        kwargs = {"np": np,
                  "ntsk": ntsk,
                  "nnd": nnd,
                  "ppnd": ppnd,
                  "tskpnd": tskpnd,
                  "pptsk": pptsk,
                  "mpp": mpp}
        return kwargs
    
    def get_slurm_parameters(self,
                             job_name: str="job",
                             partition: str="",
                             timelimit: str="",
                             e_notif: str="",
                             check_parallelization=True,
                             **kwargs) -> str:
        """Checks given parameters and returns the slurm preamble (as a string)"""

        self.para = kwargs
        cluster = self.cluster

        if cluster.upper() == "MAHTI":
            if self.para["ppnd"] == 0: self.para["ppnd"] = 128
            if partition == "": partition = "medium"
            if timelimit == "": timelimit = "36:00:00"
            csc = True
        elif cluster.upper() == "PUHTI":
            if self.para["ppnd"] == 0: self.para["ppnd"] = 40
            if self.para["mpp"] == 0: self.para["mpp"] = 8
            if partition == "": partition = "large"
            if timelimit == "": timelimit = "72:00:00"
            csc = True
        elif cluster.upper() == "OBERON":
            if self.para["ppnd"] == 0: self.para["ppnd"] = 40
        elif cluster.upper() == "PUCK":
            if self.para["ppnd"] == 0: self.para["ppnd"] = 24

        if check_parallelization:
            self.para = self.check_parallelization(**self.para)
        
        spar  = f"#!/bin/bash -l\n#SBATCH -J {job_name}\n"
        spar += f"#SBATCH -o {job_name}.out\n"
        spar += f"#SBATCH -e {job_name}.err\n"
        spar += f"#SBATCH --ntasks={self.para['ntsk']}\n"
        spar += f"#SBATCH --ntasks-per-node={self.para['tskpnd']}\n"
        spar += f"#SBATCH --nodes={self.para['nnd']}\n"
        spar += f"#SBATCH --cpus-per-task={self.para['pptsk']}\n"
                 
        if self.para["mpp"] != 0:
            spar += f"#SBATCH --mem-per-cpu={self.para['mpp']}GB\n"
        if partition != "":
            spar += f"#SBATCH --partition={partition}\n"
        if timelimit != "":
            spar += f"#SBATCH -t {timelimit}\n"
        if csc: 
            spar += f"#SBATCH --account={account}\n"
        if e_notif != "":
            spar +=  "#SBATCH --mail-type=END,FAIL\n"
            spar += f"#SBATCH --mail-user={e_notif}\n"

        return spar
    
    def set_slurm_parameters(self, parameters: str) -> None:
        if parameters == None:
            parameters = self.get_slurm_parameters()
        self.slurm = parameters

    def get_modules(self, dft: str=None, omp: bool=True, cluster: str=None, vasp_gam: bool=False) -> str:
        """Returns a string containing the commands to load the necessary modules,
        depending on the system and code used."""
        if dft == None:
            dft = self.dft.upper()
        else:
            dft = dft.upper()
        if cluster == None:
            cluster = self.cluster.upper()
        else:
            cluster = cluster.upper()

        if cluster == "MAHTI":
            if dft == "GPAW":
                modle  = "module load gpaw/25.1.0-omp\n"
                omp = True
            elif dft == "VASP":
                modle = "module load vasp/6.4.3\n"
                if self.ase:
                    modle += "\nmodule load gpaw"
                    if vasp_gam:
                        modle += "\nexport ASE_VASP_COMMAND=\"vasp_gam\"\n"
                    else:
                        modle += "\nexport ASE_VASP_COMMAND=\"vasp_std\""
                    modle += "\nexport VASP_PP_PATH=\"/scratch/project_2012891/\""
        elif cluster == "PUHTI":
            if dft == "GPAW":
                modle  = "module load mpich/3.3.1"
                modle += "\nmodule load python-env"
                modle += "\nmodule load gpaw\n"
            elif dft == "VASP":
                modle = "module load vasp/6.4.3\n"
                if self.ase:
                    modle += "\nmodule load gpaw\n"
                    if vasp_gam:
                        modle += "\nexport ASE_VASP_COMMAND=\"vasp_gam\""
                    else:
                        modle += "\nexport ASE_VASP_COMMAND=\"vasp_std\""
                    modle += "\nexport VASP_PP_PATH=\"/scratch/project_2012891/\""
        elif cluster == "OBERON":
            modle = "module load gpaw/1.5.1-gcc"
        elif cluster == "PUCK":
            modle = "module load puck_gpaw/1.5.1-ase3.17.0-gcc"

        if omp:
            modle += "\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK"
            modle += "\nexport OMP_PLACES=cores"

        return modle + "\n"

    def set_modules(self, modules: str=None) -> None:
        if modules == None:
            modules = self.get_modules()
        self.modules = modules
    
    def get_procedure(self, vasp_gam: bool=False) -> str:
        """Returns string containing the procedure (steps in the SLURM input script)"""
        par_exec = ""
        proc = ""
        
        if self.csc:
            par_exec = "srun"

            if self.dft.upper() == "GPAW":
                proc = f"gpaw-python {self.ase_pyfile}"
            elif self.dft.upper() == "VASP":
                if self.ase:
                    proc = f"python3 {self.ase_pyfile}"
                elif vasp_gam:
                    proc = f"vasp_gam"
                else:
                    proc = f"vasp_std"
        else:
            par_exec = f"mpirun -np {self.para['np']}"

            if self.dft.upper() == "GPAW":
                proc = f"gpaw-python {self.ase_pyfile}"
        
        proc = par_exec + " " + proc
        return proc

    def set_procedure(self, procedure: str=None) -> None:
        if procedure == None:
            procedure = self.get_procedure()
        self.procedure = procedure

    def write(self, file: str=""):
        """Writes contents to disk"""
        filecontents = self.slurm + "\n" \
                       + self.modules + "\n\n\n" \
                       + self.procedure
        with open(file,"w") as w:
            w.write(f"{filecontents}\n")


def write_sh(pyFile: str="", cluster: str="", dft: str="", account: str="", partition: str="", customName: str=None,
             **kwargs) -> str:   # kwargs: np: int=0, ntsk: int=0, nnd: int=0, ppnd: int=0, tskpnd: int=0, pptsk: int=1, mpp: int=0, timelimit: str=""
    
    """Creates a Slurm script file that runs the specified script with gpaw-python.
       pyFile  : Python script to run. Required.
       cluster : Name of the computing cluster [Mahti, Puhti, Oberon, or Puck].
       nnd     : Number of nodes to split the job onto [SBATCH: --nodes] Set to 0 to calculate automatically.
       mpp     : Memory per processor [] set to 0 for default/to exclude.
       e_notif : Give your email address, if you want a notification when the calcuations stop."""
    
    sh = sh_writer(dft=dft, cluster=cluster)

    sh.dft = dft.upper()
    
    pyFile = path.abspath(pyFile)

    if customName != None:
        namn = customName
    else:
        namn = pyFile
    
    if namn.__contains__("."):
        namn = namn[:namn.rfind(".")]
    
    if namn.__contains__("/"):
        job_name = namn[namn.rfind("/")+1:]
        sh.ase_pyfile = pyFile[pyFile.rfind("/")+1:]
    else:
        job_name = namn
        sh.ase_pyfile = pyFile

    shFile = path.abspath(f"{namn}.sh")

    s_params = sh.get_slurm_parameters(job_name=job_name,
                                       account=account,
                                       partition=partition,
                                       **kwargs)
    sh.set_slurm_parameters(parameters=s_params)

    modules = sh.get_modules(dft=dft,
                             omp=True,
                             cluster=cluster,
                             vasp_gam=kwargs["gamma_exec"])
    sh.set_modules(modules=modules)

    procedure = sh.get_procedure(vasp_gam=kwargs["gamma_exec"])
    sh.set_procedure(procedure=procedure)

    sh.write(shFile)
    
    if shFile.__contains__("/"):
        shFile = shFile[shFile.rfind("/")+1:]
            
    return shFile


def write_py(inFile: str="", dft: str="", template: str="./pytem.py") -> str:
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
        template = path.abspath(path.expanduser(template))
        try:
            with open(template, "r") as puh:
                for l in puh:
                    if l.__contains__("?name?"):
                        l = l.replace("?name?", str(name))
                    if l.__contains__("?inFile?"):
                        l = l.replace("?inFile?", str(inFile))
                    pytxt = pytxt + l
        except FileNotFoundError:
            pytxt = f"The template file was not found in {template}"

        with open(pyFile,"w") as w:
            w.write(f"{pytxt}\n")
        
        return pyFile
    else:
        return "_na"


def write_vasp(inFile: str="", incartem: str=None) -> str:
    """Writes input files for a Vasp calculation
    inFile: file containing atomic coordinates"""
    from lkit_vasp_parameters import incar_recipes
    from lkit_vasp_parameters import vasp_U_by_element

    wd, posFileName = path.split(path.abspath(inFile))
    write_potcar = not path.exists(path.join(wd, "POTCAR"))

    # Choose the correct set of defaults
    param = incar_recipes['LKIT']

    # Figure out which atoms are involved and in which order (could also call ltool to create POTCAR)
    if posFileName not in {"POSCAR", "CONTCAR"}:
        from ase.io import read
        atoms = read(inFile)
        ltool.write_poscar(atoms=atoms, directory=wd)
        posFileName = "POSCAR"
    
    atomtypes, _ = ltool.get_natoms_poscar(path.join(wd, posFileName))

    if "MAGMOM" in param:
        magmom = ""
        for a in atomtypes:
            if a == "Ni" or a == "Co":
                magmom = " ".join((magmom, str("1.5 ")))
            else:
                magmom = " ".join((magmom, str("0.0 ")))
        param["MAGMOM"] = magmom
    if "LDAU" in param:
        ldauu = ""
        ldaul = ""
        for a in atomtypes:
            if a in vasp_U_by_element:
                ldaul = " ".join((ldaul, str(vasp_U_by_element[a]["L"])))
                ldauu = " ".join((ldauu, str(vasp_U_by_element[a]["U"])))
            else:
                ldaul = " ".join((ldaul, str(-1)))
                ldauu = " ".join((ldauu, str(0.0)))
        ldauj = f"{'0.00 ' * len(atomtypes)}"
        param["LDAUU"] = ldauu
        param["LDAUJ"] = ldauj
        param["LDAUL"] = ldaul

    # Open incartem and read+append lines to INCAR with the correct substitution in place of *
    incar = path.join(wd, "INCAR")
    with open(incar, "w") as ic:
        with open(incartem, "r") as ict:
            for l in ict:
                for key in param:
                    if l.__contains__(key.join((" ", " "))):
                        l = l.replace("*", str(param[key]))
                if not l.__contains__("*"):
                    ic.write(l)

    if write_potcar:
        ltool.write_potcar(inFile)

    return incar



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
    gtem = path.abspath(path.join(lbfolder, config.templates["gtem"]))
    vtem = path.abspath(path.join(lbfolder, config.templates["vtem"]))
    nebtem = path.abspath(path.join(lbfolder, config.templates["nebtem"]))
    incartem = path.abspath(path.join(lbfolder, config.templates["incartem"]))
    
    arpar = argparse.ArgumentParser(prog="LBatch", 
                                    description="Creates the necessary files to run a DFT calculation through SLURM")

    arpar.add_argument("file", type=str, 
                       help="The file containing the atomic coordinates")
    arpar.add_argument("-v", action="store_true", 
                       help="Verbose. By default lbatch returns only the name of the SLURM script file")
    arpar.add_argument("-d", metavar="DFT CODE", type=str, default=dft,
                        help=f"Which DFT package to run the calculations with. Currently the options are GPAW and VASP. Defaults to \"{cluster}\"")
    arpar.add_argument("-c", metavar="CLUSTER", type=str, default=cluster, 
                        help=f"The name of the cluster the job will be run on. This allows the code to generate suitable default values. Defaults to \"{cluster}\"")
    arpar.add_argument("-a", metavar="ACCOUNT", type=str, default=account,
                        help=f"The account used for computer time when running on CSC supercomputers. Defaults to \"{account}\"")
    arpar.add_argument("-p", type=int, default=0,
                        help="Number of processors")
    arpar.add_argument("-n", type=int, default=0, 
                       help="Number of nodes")
    arpar.add_argument("-m", type=float, default=0., 
                       help="Memory per processor (GB)")
    arpar.add_argument("-t", type=int, default=0, 
                       help="Time limit in hours (int). There's no sanity checking here so check the limit for each cluster")
    arpar.add_argument("-s", action="store_true",
                       help="Automatically submit the job to the SLURM system")
    arpar.add_argument("--ase", action="store_true", 
                       help="Uses ASE the ASE/Python interface to the DFT calculator. Enabled automatically when DFT CODE = \"GPAW\"")
    arpar.add_argument("--email", type=str, default="", 
                       help="You can give an email address to recieve a notification of events during the job's execution")
    arpar.add_argument("--sin", action="store_true",
                       help="Loads the 'Python singularity' containing scipy, pandas, matplotlib etc.")
    arpar.add_argument("--ppnd", type=int, default=0, 
                       help="Processors per node")
    arpar.add_argument("--part", metavar="PARTITION", type=str, default="", 
                       help="The SLURM partition to use. Defaults to \"medium\" on Mahti and \"large\" on Puhti")
    arpar.add_argument("--sh", action="store_true", 
                       help="Generates ONLY the SLURM script file. \"file\" should then contain the script to be run, eg. a python file")
    arpar.add_argument("--neb", action="store_true", 
                       help="Generates a file to run a Nudged Elastic Band (NEB) calculation using ASE")
    arpar.add_argument("--gam", action="store_true",
                       help="Uses the vasp_gam executable instead of vasp_std. Improved performance for jobs without more extensive k-point sampling")

    args = arpar.parse_args()

    fil = args.file
    jobname = None
    dft = args.d.upper()
    use_ase = args.ase
    verbose = args.v
    lazy = not args.sh
    gam = args.gam

    timelimit = args.t
    if timelimit != 0:
        timelimit = f"{str(timelimit).rjust(2,'0')}:00:00"
    else:
        timelimit = ""
    

    # Should not include the srun / mpirun command yet
    run_cmd = ""

    if args.neb:
        use_ase = True
        pytem = nebtem
    elif dft == "GPAW":
        use_ase = True
        pytem = gtem
    elif dft == "VASP":
        if use_ase:
            pytem = vtem

    
    # Lazy mode writes the Python file for ASE on the user's behalf.
    # I'm lazy so it's set as default. Can be disabled by using --sh
    if lazy:
        if use_ase:
            if verbose: print("Python file:")
            _fil = write_py(inFile=fil, dft=dft, template=pytem)
            if _fil == "_na":
                print(f"No file \"{fil}\" found")
            elif verbose:
                print(_fil)
                print("It is recommended that you check the parameters before "+ \
                      "submitting")
            run_cmd += fil
        elif dft == "VASP":
            _fil = write_vasp(inFile=fil, incartem=incartem)
            jobname = "vasp"
        else:
            raise NotImplementedError
    else:
        _fil = fil

    fil = path.abspath(_fil)


    if path.exists(fil):
        sph = write_sh(pyFile=fil, cluster=args.c, dft=dft,
                       e_notif=args.email, account=args.a, partition=args.part,
                       np=args.p, nnd=args.n, ppnd=args.ppnd, mpp=args.m,
                       timelimit=timelimit, customName=jobname, gamma_exec=gam)
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
    
    
