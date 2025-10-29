
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read, write, Trajectory
from ase.optimize import FIRE
from ase.parallel import paropen, world
from gpaw import GPAW, FermiDirac, RMMDIIS, PoissonSolver, Mixer
from datetime import datetime, timedelta
import numpy as np
from ase.neb import NEB, NEBTools


#####
##########
###############

filename = "?name?"
inF = f"?name?.?inExt?"
colF = "neb.txt"

_gpts = (64, 72, 128)

_neb_fmax = 0.1
_cineb_fmax = 0.02

_neb = True
_cineb = True

_neb_k = 0.5
_cineb_k = 0.5

###############
##########
#####


t_start = datetime.now()

msg = ""
f_end = "n.a."


in_traj = read(inF)

initial = in_traj[0]
final = in_traj[-1]

img_n = len(in_traj) - 2

rank, size = world.rank, world.size
n = size // img_n     # number of cpu's per image
j = 1 + rank // n  # my image number
assert img_n * n == size



   ###
 ###-###
##-----##

images = [initial]

i = 0
for img in in_traj[1:-1]:
    ranks = np.arange(i * n, (i + 1) * n)
    
    if rank in ranks:
        calc = GPAW(gpts=_gpts,
            occupations=FermiDirac(width=0.1),
            xc='BEEF-vdW',
            eigensolver=RMMDIIS(niter=3),
            nbands=-500,
            setups={'Zr': ':d,2.0'},
            maxiter=2000,
            txt="neb_relax_img%d.txt" % ( i+1 ),
            communicator=ranks,
            mixer=Mixer(0.1, 10, weight=50.0),
            # convergence={'eigenstates': 1.0e-6,'density': 1.0e-5,'energy': 1.0e-5},
            poissonsolver=PoissonSolver(eps=1e-8))

        img.set_calculator(calc)
    i += 1

    images.append(img)

images.append(final)


lastSavedTo = ""

if _neb:
    neb = NEB(images,k=_neb_k, climb=False, parallel=True)
    qn = FIRE(neb, logfile=filename+'.log')
    qn.attach(Trajectory(f"{filename}_neb.traj", 'a', neb))
    lastSavedTo = f"{filename}_neb.traj"
    qn.run(fmax=_neb_fmax)
    images = neb.images

if _cineb:
    cineb = NEB(images,k=_cineb_k, climb=True, parallel=True)
    qn = FIRE(cineb, logfile=filename+'.log')
    qn.attach(Trajectory(f"{filename}_cineb.traj", 'a', cineb))
    lastSavedTo = f"{filename}_cineb.traj"
    qn.run(fmax=_cineb_fmax)
    images = cineb.images

##-----##
 ###-###
   ###



if rank == 0:
    # Get energies for images and save the initial and final
    images = read(f"{lastSavedTo}@-{img_n+2}:")

    initial = images[0]
    final = images[-1]

    Es_zp = [e.get_potential_energy() for e in images]

    e_zp_initial = Es_zp[0]
    e_zp_final = Es_zp[-1]

    tools = NEBTools(images)

    ind_TS = Es_zp.index(max(Es_zp))
    TS = images[ind_TS]

    e_zp_TS = tools.get_barrier(raw=True)[0]

    Ea_f = e_zp_TS - e_zp_initial
    Ea_r = e_zp_TS - e_zp_final
    dE = e_zp_final - e_zp_initial

    f_end = tools.get_fmax()


    # Save (CI)NEB plot (.svg)
    fig = tools.plot_band()
    fig.savefig(f"{filename}.svg",dpi=300, format="svg")


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

        col.write(f"\n    Job {filename}:\n")
        col.write(f"Completed at {nu.isoformat(timespec='seconds').replace('T',' at ')} ({t_D})\n")
        col.write(f"{msg}Maximum residual force: {str(round(f_end, 5))} || Grid points: {gspacing}\n\n")

        # Initial structure
        col.write(f"   Initial stucture\n")
        col.write(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(e_zp_initial, 5)).ljust(12)} eV\n")

        # Transition state
        col.write(f"   Transition state [{ind_TS}]\n")
        col.write(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(e_zp_TS, 5)).ljust(12)} eV\n")
        col.write(f"Activation energy:\n{'  Forward:'.ljust(indent)} {str(round(Ea_f, 5)).ljust(12)} eV\n{'  Reverse:'.ljust(indent)} {str(round(Ea_r, 5)).ljust(12)} eV\n")
        col.write(f"{'  dE:'.ljust(indent)} {str(round(dE, 5)).ljust(12)} eV\n")


        # Final structure
        col.write(f"   Final stucture\n")
        col.write(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(e_zp_final, 5)).ljust(12)} eV\n")

        col.write(f"\n{'='*100}\n")

    """------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""
    """------_______------""""""------_______------""""""------_______------""""""------_______------""""""------_______------"""
