
import os.path as path
import subprocess
import ltool
from config import config

class ShWriter:
    """Class to store functions needed for creating, checking, and
    writing SLURM batch scripts
    """
    def __init__(self,
                 job_name: str,
                 dft: str,
                 cluster: str = "Mahti",
                 omp: bool = True,
                 ase_pyfile: str = None,
                 ):
        self.job_name = job_name
        self.dft = dft.upper()
        self.cluster = cluster.upper()
        self.omp = omp
        self.ase_pyfile = ase_pyfile
        self.vasp_pp_path = "/scratch/project_2012891/"
        self.csc: bool = True
        self.ase: bool = True

        self.slurm_lines: str = None
        self.module_lines: str = None
        self.procedure: str = None

    def get_slurm_lines(self) -> list[str]:
        assert self.params != None, "SLURM parameters not set"
        spar  = [
            f"#!/bin/bash -l\n#SBATCH -J {self.job_name}",
            f"#SBATCH -o {self.job_name}.out",
            f"#SBATCH -e {self.job_name}.err",
            f"#SBATCH --ntasks={self.params['ntsk']}",
            f"#SBATCH --ntasks-per-node={self.params['tskpnd']}",
            f"#SBATCH --nodes={self.params['nnd']}",
            f"#SBATCH --cpus-per-task={self.params['pptsk']}",
        ]
        
        if self.params["mpp"] != 0:
            spar.append(f"#SBATCH --mem-per-cpu={self.params['mpp']}GB")
        if self.params['partition'] != "":
            spar.append(f"#SBATCH --partition={self.params['partition']}")
        if self.params['time_limit'] != "":
            spar.append(f"#SBATCH -t {self.params['time_limit']}")
        if self.csc: 
            spar.append(f"#SBATCH --account={self.params['account']}")
        if self.params['email_notif'] != "":
            spar.append("#SBATCH --mail-type=END,FAIL")
            spar.append(f"#SBATCH --mail-user={self.params['email_notif']}")

        return spar
    
    def set_slurm_lines(self, slurm_lines: list[str] = None) -> None:
        if slurm_lines == None:
            slurm_lines = self.get_slurm_lines()
        self.slurm_lines = slurm_lines
    
    def check_parallelization(self, 
                              np: int = 0,
                              ntsk: int = 0,
                              nnd: int = 0,
                              ppnd: int = 4,
                              tskpnd: int = 0,
                              pptsk: int = 1,
                              mpp: int = 0,
                              **_
                              ) -> dict:
        """Checks and adjusts the parallellization values that have not
        been specified so that they will be functional. Note that very
        minimal optimization is being done here, meaning the resources
        may end up being distributed in an suboptimal way if the values
        requested are not sensible.
        
        Parameters:

        np: int
            Number of processors (in total)
        ntsk: int
            Number of MPI tasks (in total)
        nnd: int
            Number of compute nodes
        ppnd: int
            Number of processors per compute node
        tskpnd: int
            Number of MPI tasks per compute node
        pptsk: int
            Number of processors per MPI task
        mpp: int
            Memory per processor (in GB)

        Returns:

        kwargs: dict
            Dictionary containing the adjusted parameters
        """
        if ppnd <= 8:
            print("The number of processors per node is very low.",
                  "Check that everything is as you wanted.")

        if tskpnd > 0:
            if pptsk == 1 or tskpnd * pptsk > ppnd:
                # If the provided number of tasks and processors
                # per task exceeds the processors in the node, we
                # prioritize tasks and recalculate no. threads
                pptsk = ppnd // tskpnd
            if nnd != 0 and ntsk != nnd * tskpnd:
                ntsk = nnd * tskpnd
                print("\033[93mThe total number of MPI tasks has been changed\033[0m")
            elif nnd == 0: 
                if np > 0:
                    ntsk = np // tskpnd
                    print("\033[93mThe total number of MPI tasks has been changed\033[0m")
                else:
                    ntsk = tskpnd
                    print("\033[93mThe total number of MPI tasks has been changed\033[0m")
        else:
            # Adjust tasks per node to fit processors per task (default pptsk=1).
            tskpnd = ppnd // pptsk
            if nnd == 0:
                if ntsk == 0 and np > 0:
                    while ntsk * pptsk < np:
                        ntsk = ntsk + 1
            else:
                ntsk = tskpnd * nnd
        if ntsk > 0 and nnd > 0:
            if (nnd-1) * tskpnd >= ntsk:
                print(("\033[93mYou've requested an excessive amount of nodes for"
                      " the tasks you want.\033[0m This may be a poor use of resources."))

        if nnd == 0:
            nnd = 1
            if tskpnd != 0: 
                # If the number of tasks per node is specified (should
                # always be), add nodes until the requested number of
                # tasks is reached
                while nnd * tskpnd < ntsk:
                    nnd = nnd + 1
        # Warning: if the user somehow specifies a number of nodes but
        # an impossible amount of tasks, this will not fix it. But
        # let's just call that user error to avoid choosing which to
        # prioritize.
        assert nnd * tskpnd >= np, \
            ("The specified options require more"
            " processors than will fit in the number of"
            " nodes requested")

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
                  "mpp": mpp,
                  **_}
        return kwargs
    
    def set_slurm_parameters(self,
                             check_parallelization=True,
                             **kwargs) -> None:
        """Checks given parameters and returns the slurm preamble as a
        list of strings (lines)
        """
        self.params = kwargs
                             
        if self.cluster.upper() == "MAHTI":
            if self.params["ppnd"] == 0: self.params["ppnd"] = 128
            if self.params['partition'] == "": self.params['partition'] = "medium"
            if self.params['time_limit'] == "": self.params['time_limit'] = "36:00:00"
            self.csc = True
        elif self.cluster.upper() == "PUHTI":
            if self.params["ppnd"] == 0: self.params["ppnd"] = 40
            if self.params["mpp"] == 0: self.params["mpp"] = 8
            if self.params['partition'] == "": self.params['partition'] = "large"
            if self.params['time_limit'] == "": self.params['time_limit'] = "72:00:00"
            self.csc = True
        elif self.cluster.upper() == "OBERON":
            if self.params["ppnd"] == 0: self.params["ppnd"] = 40
        elif self.cluster.upper() == "PUCK":
            if self.params["ppnd"] == 0: self.params["ppnd"] = 24

        if check_parallelization:
            self.params = self.check_parallelization(**self.params)

    def prepare_slurm(self, **kwargs) -> None:
        self.set_slurm_parameters(**kwargs)
        lines = self.get_slurm_lines()
        lines.append("")  # An empty line after the SLURM block for prettiness
        self.set_slurm_lines(lines)

    def get_modules(self) -> list[str]:
        """Returns a list of strings (lines) containing the commands
        to load the necessary modules, depending on the system and DFT
        code used.
        """
        modle = []

        if self.cluster == "MAHTI":
            if self.dft.upper() in ["GPAW", "SJM"]:
                modle = ["module load gpaw/25.1.0-omp"]
                self.omp = True
            elif self.dft.upper() == "VASP":
                modle = ["module load vasp/6.4.3"]
                if self.ase:
                    modle.append("module load gpaw")
                    modle.append(f"export VASP_PP_PATH={self.vasp_pp_path}")
        elif self.cluster == "PUHTI":
            if self.dft.upper() in ["GPAW", "SJM"]:
                modle = [
                    "module load mpich/3.3.1",
                    "module load python-env",
                    "module load gpaw/24.6.0-omp"
                    ]
                self.omp = True
            elif self.dft.upper() == "VASP":
                modle = ["module load vasp/6.4.3"]
                if self.ase:
                    modle.extend([
                        "module load gpaw",
                        f"export VASP_PP_PATH={self.vasp_pp_path}"
                        ])
        elif self.cluster == "OBERON":
            modle = ["module load gpaw/1.5.1-gcc"]
        elif self.cluster == "PUCK":
            modle = ["module load puck_gpaw/1.5.1-ase3.17.0-gcc"]

        if self.omp:
            modle.extend([
                "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
                "export OMP_PLACES=cores"
                ])

        return modle

    def set_modules(self, module_lines: list[str] = None) -> None:
        if module_lines == None:
            module_lines = self.get_modules()
        self.module_lines = module_lines

    def prepare_modules(self, **kwargs) -> None:
        module_lines = self.get_modules(**kwargs)
        module_lines.extend(("", ""))  # Add two empty lines
        self.set_modules(module_lines)
    
    def get_procedure(self, vasp_gam: bool = False) -> list[str]:
        """Returns string containing the procedure (steps in the SLURM input script)"""
        parallel_command = ""
        procedure = []
        
        if self.dft.upper() == "VASP":
            if vasp_gam:
                procedure.append("export ASE_VASP_COMMAND='vasp_gam'")
            else:
                procedure.append("export ASE_VASP_COMMAND='vasp_std'")

        if self.csc:
            parallel_command = "srun"

            if self.dft.upper() in ["GPAW", "SJM"]:
                executable = f"gpaw-python {self.ase_pyfile}"
            elif self.dft.upper() == "VASP":
                if self.ase:
                    executable = f"python3 {self.ase_pyfile}"
                elif vasp_gam:
                    executable = f"vasp_gam"
                else:
                    executable = f"vasp_std"
            else:
                raise NotImplementedError
        else:
            parallel_command = f"mpirun -np {self.params['np']}"
            if self.dft.upper() in ["GPAW", "SJM"]:
                executable = f"gpaw-python {self.ase_pyfile}"
        
        procedure.append(parallel_command + " " + executable)
        return procedure

    def set_procedure(self, procedure: list[str] = None) -> None:
        if procedure == None:
            procedure = self.get_procedure()
        self.procedure = procedure

    def prepare_procedure(self, **kwargs) -> None:
        procedure = self.get_procedure(**kwargs)
        self.set_procedure(procedure)

    def write(self, file: str="") -> None:
        """Writes contents to file"""
        with open(file,"w") as w:
            for l in self.slurm_lines + self.module_lines + self.procedure:
                w.write(l + "\n")


def write_sh(
        pyFile: str = "",
        cluster: str = "",
        dft: str = "",
        account: str = "",
        partition: str = "",
        job_name: str = None,
        gamma_exec: bool = True,
        **kwargs
        ) -> str:
    
    """Creates a Slurm script file that runs the specified script with (gpaw-)python.

    pyFile : str
        Python script to run. Required.
    cluster : str
        Name of the computing cluster [Mahti, Puhti, Oberon, or Puck].
    nnd : int
        Number of nodes to split the job onto [SBATCH: --nodes] Set to 0 to calculate automatically.
    mpp : int | float
        Memory per processor [] set to 0 for default/to exclude.
    email_notif : str
        Email address for sending a notification when the calcuations stop. Empty string or None means no notification
    """

    pyFile = path.abspath(pyFile)

    if job_name != None:
        namn = job_name
    else:
        namn = pyFile
    
    if namn.__contains__("."):
        namn = namn[:namn.rfind(".")]
    
    if namn.__contains__("/"):
        job_name = namn[namn.rfind("/")+1:]
        ase_pyfile = pyFile[pyFile.rfind("/")+1:]
    else:
        job_name = namn
        ase_pyfile = pyFile
    
    sh = ShWriter(job_name=job_name,
                  dft=dft,
                  cluster=cluster,
                  omp=True,
                  ase_pyfile=ase_pyfile,
                  )

    shFile = path.abspath(f"{namn}.sh")

    sh.prepare_slurm(
        account=account,
        partition=partition,
        **kwargs
    )
    sh.prepare_modules()
    sh.prepare_procedure(vasp_gam=gamma_exec)
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
    dft = "SJM"
    lbfolder = path.dirname(path.realpath(__file__))
    gtem = path.abspath(path.join(lbfolder, config.templates["gtem"]))
    vtem = path.abspath(path.join(lbfolder, config.templates["vtem"]))
    nebtem = path.abspath(path.join(lbfolder, config.templates["nebtem"]))
    sjmtem = path.abspath(path.join(lbfolder, config.templates["sjmtem"]))
    incartem = path.abspath(path.join(lbfolder, config.templates["incartem"]))
    
    arpar = argparse.ArgumentParser(prog="LBatch", 
        description=("Creates the necessary files to submit and run a DFT"
                    " calculation using SLURM."))

    arpar.add_argument("file", type=str, 
                       help="The file containing the atomic coordinates")
    arpar.add_argument("-v", action="store_true", 
                       help=("Verbose. By default lbatch returns only the name"
                            " of the SLURM script file"))
    arpar.add_argument("-d", metavar="DFT CODE", type=str, default=dft,
                        help=("Which DFT package to run the calculations with."
                             " Currently the options are 'GPAW', 'SJM' and 'VASP'"))
    arpar.add_argument("-c", metavar="CLUSTER", type=str, default=cluster, 
                        help=("The name of the cluster the job will be run on."
                             f" This allows the code to generate suitable default values. Defaults to '{cluster}'"))
    arpar.add_argument("-a", metavar="ACCOUNT", type=str, default=account,
                        help=("The account used for computer time when running"
                             " on CSC supercomputers. Defaults to"
                            f" '{account}'"))
    arpar.add_argument("-p", type=int, default=0,
                        help=("Number of processors. In practice, you may want"
                             " to specify number of nodes and processors per"
                             " node instead"))
    arpar.add_argument("-n", type=int, default=0, 
                       help="Number of nodes")
    arpar.add_argument("-m", type=float, default=0., 
                       help="Memory per processor (GB)")
    arpar.add_argument("-t", type=int, default=0, 
                       help="Time limit in hours (int). There's no sanity checking here so check the limit for each cluster")
    arpar.add_argument("-s", action="store_true",
                       help="Automatically submit the job to the SLURM system")
    arpar.add_argument("--ase", action="store_true", 
                       help="Uses ASE the ASE/Python interface to the DFT calculator. Enabled automatically when DFT CODE = 'GPAW' or 'SJM'")
    arpar.add_argument("--email", type=str, default="", 
                       help="You can give an email address to recieve a notification of events during the job's execution")
    arpar.add_argument("--ppnd", type=int, default=0, 
                       help="Processors per node")
    arpar.add_argument("--tskpnd", type=int, default=0, 
                       help="MPI tasks per node")
    arpar.add_argument("--part", metavar="PARTITION", type=str, default="", 
                       help="The SLURM partition to use. Defaults to 'medium' on Mahti and 'large' on Puhti")
    arpar.add_argument("--sh", action="store_true", 
                       help="Generates ONLY the SLURM script file. 'file' should then contain the script to be run, eg. a python file")
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

    time_limit = args.t
    if time_limit != 0:
        time_limit = f"{str(time_limit).rjust(2,'0')}:00:00"
    else:
        time_limit = ""
    

    # Should not include the srun / mpirun command yet
    run_cmd = ""

    if args.neb:
        use_ase = True
        pytem = nebtem
    elif dft == "GPAW":
        use_ase = True
        pytem = gtem
    elif dft == "SJM":
        use_ase = True
        pytem = sjmtem
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
                print(f"No file '{fil}' found")
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
        sph = write_sh(pyFile=fil,
                       cluster=args.c,
                       dft=dft,
                       email_notif=args.email,
                       account=args.a,
                       partition=args.part,
                       np=args.p,
                       nnd=args.n,
                       ppnd=args.ppnd,
                       tskpnd=args.tskpnd,
                       mpp=args.m,
                       time_limit=time_limit,
                       job_name=jobname,
                       gamma_exec=gam)
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
        print(f"No file '{fil}' found")
    
    
