
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read, write, Trajectory
from ase.optimize import FIRE
from ase.parallel import paropen, world
from ase.vibrations import Infrared
from ase.calculators.vasp import Vasp
from datetime import datetime
import numpy as np
from os import remove as os_rm
import os.path as path


#####
##########
###############

name = "?name?"
inF = "?inFile?"
colF = "energies.txt"

_encut = 400
_nelm = 400
_pbc = True
_kpts = (2,2,1)
_kpar = 4

_vdw = 12

_opt = True
_opt_ase = False
_ibrion = 2
_fix = True
_zfix = 21.0
_fmax = 0.01
_nelm_opt = 100

_cube = True
_wf = False
_ir = False
_irInd = []

###############
##########
#####


t_start = datetime.now()
atoms = read(inF)
atoms.set_pbc(_pbc)

if _fix:
    mask = [a.position[2] < _zfix for a in atoms]
    c= FixAtoms(mask=mask)
    atoms.set_constraint(c)
else:
    atoms.set_constraint()

msg = ""
f_end = "n.a."


if _opt:
    _nelm = _nelm_opt
if _opt_ase or not _opt:
    _ibrion = -1

if _cube:
    _lcharg = True
else:
    _lcharg = False

   ###
 ###-###
##-----##

calc = Vasp(
    xc = "PBE",
    encut = _encut,             # PW cutoff energy
    kpts = _kpts,
    # nbands = 3000,            # Number of (empty) bands in the calculation
    ediff = 5e-5,               # SC convergence
    nelmin = 8,                 # Min. SCF steps
    nelm = _nelm,               # Max. SCF steps
    istart = 0,                 # 0 for normal; 1 for restart
    algo = "Normal",            # electronic optimization algorithm. 'Normal'=block Davidson; 'All' may be better for hybrids
    prec = "Accurate",          # Sets accurate defaults. use only if lreal
    ismear = 1,                 # Fermi smearing
    sigma = 0.1,                # Smearing width
    ispin = 1,                  # 2 = Spin polarized calculation
    isym = 0,                   # Does not use symmetry
    ivdw = _vdw,                # Grimme D3 correction
    # Hubbard U from Materials Project:
    # https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/hubbard-u-values
    ldau = True,                # Use LDA+U
    ldautype = 2,               # LDA+U according to Dudarev et al.
    ldau_luj = {'Mo': {'L': 2, 'U': 4.4, 'J': 0},
              'Ni': {'L': 2, 'U': 6.2, 'J': 0}},
    lmaxmix = 6,                # Maximum number of steps stored in the Broyden mixer
    ## GEOMETRY OPTIMIZATION 
    ibrion = _ibrion,           # Structure optimization: ibrion=1-3
    nsw = 500,                  # Max number of relaxation steps
    ediffg = -_fmax,            # Criterion for GO convergence. Positive: energy, negative: force
    ## PROJECTIONS / PAWS 
    lreal = "Auto",             # Real-space determination of projections
    setups = {'Mo': '_sv_GW',
              'S': '_GW',
              'Ag': '_GW',
              'Ni': '_sv_GW',
              'H': '_GW',
              'O': '_GW'},
    lasph = True,               # Non-spherical contributions in PAW spheres
    ## PARALLELIZATION 
    # npar = 4,                 # Number of parallel bands
    kpar = _kpar,               # k-point parallelization
    ## OUTPUT 
    txt = f"{name}.out",
    lorbit = 11,                # Saves PROCAR file with PDOS
    lcharg = _lcharg,           # Saves files with charge densities
    )



atoms.calc = calc


if _opt and _opt_ase:
    traj = Trajectory(f"{name}.traj", "a", atoms)
    
    dyn = FIRE(atoms, logfile=f"{name}.log")
    dyn.attach(traj)
    dyn.run(fmax=_fmax)
    optdone = True
    
    f_end = round(np.apply_along_axis(np.linalg.norm, 1, atoms.get_forces()).max(), 5)
    

E_tot = atoms.get_total_energy()
E_zp = atoms.get_potential_energy()
E_fc = atoms.get_potential_energy(force_consistent=True)

m_mom = atoms.get_magnetic_moment()

##-----##
 ###-###
   ###



if _wf: calc.write(filename=f"{name}.gpw", mode="all")


if _ir:
    try:
        if path.exists(f"{name}_ir.all.pckl") and world.rank == 0: os_rm(f"{name}_ir.all.pckl")
        ir = Infrared(atoms,
            indices = _irInd,
            delta = 0.02,
            name = f"{name}_ir",
            nfree = 4)
        ir.run()
        ir.summary(method="frederiksen", log=f"{name}_ir.log")
        msg += f"IR analysis saved in {name}_ir.log\n"
        ir.write_spectra(out=f"{name}_spectrum.dat")
        for i in range(3*len(_irInd)):
            ir.write_mode(n=i)
        if world.rank == 0: ir.combine()
    except Exception as exc:
        with paropen(f"{name}_ir.log", "a") as fil:
            fil.write(f"Failed\n{exc}")
        raise



"""------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""
"""------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""

nu = datetime.now()
t_delta = nu - t_start
t_D = ""
t_day, t_hour, t_min, t_sec = t_delta.days, 0, t_delta.seconds // 60, t_delta.seconds % 60
if t_min >= 60: 
    t_hour, t_min = t_min // 60, t_min % 60
if t_day != 0: t_hour += t_day * 24

if t_hour > 0: t_D += f"{t_hour} h, "
if t_min > 0 or t_hour > 0: t_D += f"{t_min} min, "
t_D += f"{t_sec} s"

with paropen(colF, "a") as col:
    indent = 18

    col.write(f"\n    Job {name}:\n")
    col.write(f"Completed at {nu.isoformat(timespec='seconds').replace('T',' at ')} ({t_D})\n")
    if f_end != "n.a.": f_end = str(round(f_end, 5)).replace('.',',')
    col.write(f"{msg}Maximum residual force: {f_end} || Magnetic moment: {round(m_mom, 5)}\n")
    col.write(f"{'Total energy:'.ljust(indent)} {str(round(E_tot, 5)).replace('.',',').ljust(12)} eV\n")
    col.write(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(E_zp, 5)).replace('.',',').ljust(12)} eV\n")
    col.write(f"{'Force-consistent:'.ljust(indent)} {str(round(E_fc, 5)).replace('.',',').ljust(12)} eV\n")

    col.write(f"\n{'='*100}\n")

"""------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""
"""------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""
