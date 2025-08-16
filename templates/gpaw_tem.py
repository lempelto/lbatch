
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read, write, Trajectory
from ase.optimize import FIRE
from ase.parallel import paropen, world
from ase.vibrations import Infrared
from gpaw import GPAW, FermiDirac, Davidson, PoissonSolver, MixerDif
from datetime import datetime
import numpy as np
from os import remove as os_rm
import os.path as path


#####
##########
###############

name = "?name?"
inF = "?inFile?"
colF = "results.txt"

_opt = False
_fix = True
_zfix = 9.0

_h = .2
_kpts = (1, 1, 1)
_charge = 0
_pbc = (True, True, False)
_fmax = 0.005

_cube = False
_wf = False
_ir = False
_irInd = [96, 97, 98, 99, 100, 101, 102, 103, 104]

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



   ###
 ###-###
##-----##

calc = GPAW(
    mode        = 'fd',
    spinpol     = True,
    charge      = _charge,
    h           = _h,
    kpts        = {'size': _kpts, 'gamma': True},
    xc          = 'BEEF-vdW',
    txt         = f"{name}.txt",
    occupations = FermiDirac(width=0.1),
    eigensolver = Davidson(niter=3),
    maxiter     = 5000,
    # nbands      = -500,
    setups      = {'Ni': ':d,5.5'},
    mixer       = MixerDif(0.1, 10, weight=50.0),
    # convergence = {'energy': 5.e-4, 'density': 1.e-4, 'eigenstates': 4.e-8, 'bands': 'occupied'},
    # poissonsolver = PoissonSolver(eps=2e-10),
)

atoms.calc = calc


if _opt:
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


if _cube:
    from ase.units import Bohr
    density = calc.get_all_electron_density() * Bohr**3
    write(f"{name}.cube", atoms, data=density)


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

gspacing = calc.get_number_of_grid_points()

with paropen(colF, "a") as col:
    indent = 18

    col.write(f"\n    Job {name}:\n")
    col.write(f"Completed at {nu.isoformat(timespec='seconds').replace('T',' at ')} ({t_D})\n")
    if f_end != "n.a.": f_end = str(round(f_end, 5)).replace('.',',')
    col.write(f"{msg}Maximum residual force: {f_end} || Grid points: {gspacing} || Magnetic moment: {round(m_mom, 5)}\n")
    col.write(f"{'Total energy:'.ljust(indent)} {str(round(E_tot, 5)).replace('.',',').ljust(12)} eV\n")
    col.write(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(E_zp, 5)).replace('.',',').ljust(12)} eV\n")
    col.write(f"{'Force-consistent:'.ljust(indent)} {str(round(E_fc, 5)).replace('.',',').ljust(12)} eV\n")

    col.write(f"\n{'='*100}\n")

"""------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""
"""------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""
