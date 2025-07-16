
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

from ase.units import Pascal, m

from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)


#####
##########
###############

name        = "?name?"

inF         = f"?inFile?"
colF        = "results.txt"

_opt        = False
_fix        = True
_zfix       = 9.0

_h          = .2
_kpts       = (2, 2, 1)
_spinpol    = True
_charge     = +1
_temp       = 298.15
_pbc        = (True, True, False)
_fmax       = 0.005

_vdW_radii  = {'H': 1.00,
               'C': 1.70,
               'O': 1.55,
               'S': 1.80,
               'Ni': 1.95,
               'Mn': 2.00}
def vdW_radii(atoms):
    return np.array(
        [_vdW_radii.get(symbol)
         for symbol in atoms.symbols], dtype=float)

_wf         = True
_cube       = False
_ir         = False
_irInd      = [96, 97, 98, 99, 100, 101, 102, 103, 104]

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
    
if _spinpol:
    _magmoms = [a.tag * 1.8 for a in atoms]
    atoms.set_initial_magnetic_moments(_magmoms)

msg = ""
f_end = "n.a."



## Setting up the SJM
sj = {
    'excess_electrons': -_charge,
    'grand_output': False
}

epsinf = 78.36  # dielectric constant of water at 298 K [J. Chem. Phys. 141, 174108 (2014)]
gamma = 18.4 * 1e-3 * Pascal * m
cavity = EffectivePotentialCavity(
    effective_potential = SJMPower12Potential(atomic_radii = vdW_radii,
                                              H2O_layer = True,
                                              unsolv_backside = True),
    temperature = _temp,
    surface_calculator = GradientSurface())
dielectric = LinearDielectric(epsinf=epsinf)
interactions = [SurfaceInteraction(surface_tension=gamma)]
###


   ###
 ###-###
##-RUN-##

calc = SJM(
    mode        = 'fd',
    spinpol     = _spinpol,
    # charge      = _charge,
    h           = _h,
    kpts        = _kpts,
    xc          = 'BEEF-vdW',
    txt         = f"{name}.txt",
    occupations = FermiDirac(width=0.1),
    # eigensolver = Davidson(niter=3),
    maxiter     = 5000,
    # nbands      = -500,
    setups      = {'Ni': ':d,5.5'},
    # mixer       = MixerDif(0.1, 10, weight=50.0),
    # convergence = {'energy': 5.e-4, 'density': 1.e-4, 'eigenstates': 4.e-8, 'bands': 'occupied'},
    sj          = sj,
    cavity      = cavity,
    dielectric  = dielectric,
    interactions= interactions
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

##-----##
 ###-###
   ###


E_zp = atoms.get_potential_energy()
# E_fc = atoms.get_potential_energy(force_consistent=True)

m_mom = atoms.get_magnetic_moment()
e_pot = atoms.calc.get_electrode_potential()
calc.write_sjm_traces(style='cube')


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
# # # # #                                                  OUTPUT                                                   # # # # #
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
    col.write(f"{msg}Maximum residual force: {f_end} || Grid points: {gspacing}\n")
    col.write(f"Magnetic moment: {round(m_mom, 5)} || Electrode potential: {round(e_pot, 5)}\n")
    col.write(f"{'Total energy:'.ljust(indent)} {str(round(E_tot, 5)).replace('.',',').ljust(12)} eV\n")
    col.write(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(E_zp, 5)).replace('.',',').ljust(12)} eV\n")
    # col.write(f"{'Force-consistent:'.ljust(indent)} {str(round(E_fc, 5)).replace('.',',').ljust(12)} eV\n")

    col.write(f"\n{'='*100}\n")
