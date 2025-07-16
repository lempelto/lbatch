
import argparse
import os
from lkit_vasp_parameters import vasp_pot_collections


def print_results(results: dict) -> None:
    """Formats and prints a card showing the results in 'results'."""
    import numpy as np
    f_end = None
    E_tot = None
    E_fc = None
    magmom = None

    if 'forces' in results:
        forces = results['forces']
        f_end = round(np.linalg.norm(forces, axis=1).max(), 3)
    if 'energy' in results:
        E_tot = results['energy']
    if 'free_energy' in results:
        E_fc = results['free_energy']
    if 'magmom' in results:
        magmom = round(results['magmom'], 3)

    indent = 18
    if f_end or magmom: print(f"\n    Maximum residual force: {f_end} eV/Å | Magnetic moment: {magmom}")
    if E_tot: print(f"{'    Total energy:'.ljust(indent)} {str(round(E_tot, 5)).ljust(12)} eV")
    if E_fc: print(f"{'    Force-consistent energy:'.ljust(indent)} {str(round(E_fc, 5)).ljust(12)} eV\n")


def show_results(filename: str=None, open_gui: bool=False, args=None) -> None:
    """Opens an output file (using ASE) and prints a summary of its results. Optionally opens the structure in the ASE gui"""
    from ase.io import read

    if args:
        filename = args.file
        open_gui = args.o

    atoms = read(filename)

    if atoms.calc:
        calc = atoms.calc
    else:
        # SOMETHING MORE ELEGANT HERE PLEASE
        raise Exception

    results = calc.results
    print_results(results=results)

    if open_gui:
        from ase.visualize import view
        view(atoms)


def get_natoms_poscar(poscar: str=None) -> tuple[list]:
    """Reads POSCAR (or compatible) file and returns two lists: atom types in order and their counts in order
       eg. [H, O], [2, 1]"""

    with open(poscar, 'r') as pos_stream:
        head = [pos_stream.readline() for i in range(5)]
        atomtypes = pos_stream.readline().split()
        _natoms = pos_stream.readline().split()
    natoms = [int(i) for i in _natoms]

    return atomtypes, natoms


def write_potcar(poscar: str=None, potset: str='VASP', args=None) -> None:
    if args:
        poscar = args.file
        potset = args.p

    poscar = os.path.abspath(poscar)
    wd = os.path.dirname(poscar)
    potcar = os.path.join(wd, 'POTCAR')

    pp_path = os.environ["VASP_PP_PATH"]
    pot_category = 'potpaw_PBE'
    pot_type = vasp_pot_collections[potset.upper()]

    atomtypes, _ = get_natoms_poscar(poscar=poscar)
    
    with open(potcar, 'w') as pot_stream:
        for at in atomtypes:
            if at in pot_type:
                pot_suffix = pot_type[at]
            elif at in vasp_pot_collections['VASP']:
                pot_suffix = vasp_pot_collections['VASP'][at]
            else:
                pot_suffix = ''
            paw_potcar = os.path.join(pp_path, pot_category, at+pot_suffix, 'POTCAR')
            with open(paw_potcar, 'r') as paw_stream:
                for line in paw_stream:
                    pot_stream.write(line)


def nebtool(filename: str=None, open_gui: bool=False, nimages: int=None, args=None) -> None:
    from ase.io import read

    if args:
        filename = args.file
        open_gui = args.o
        nimages = args.nimages

    images = read(f"{filename}@-{nimages}:")

    # Get energies for images and save the initial and final
    Es_zp = [e.get_potential_energy() for e in images]
    e_zp_initial = Es_zp[0]
    e_zp_final = Es_zp[-1]

    ind_TS = Es_zp.index(max(Es_zp))
    TS = images[ind_TS]

    from ase.neb import NEBTools
    tools = NEBTools(images)
    e_zp_TS = tools.get_barrier(raw=True)[0]

    Ea_f = e_zp_TS - e_zp_initial
    Ea_r = e_zp_TS - e_zp_final
    dE = e_zp_final - e_zp_initial

    f_end = tools.get_fmax()

    indent = 18
    print(f"\n{'='*100}")
    print(f"==  Job {filename}:\n")
    print(f"Maximum residual force: {str(f_end).replace('.',',')}\n")
    # Initial structure
    print(f"   Initial stucture")
    print(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(e_zp_initial, 5)).replace('.',',').ljust(12)} eV\n")
    # Transition state
    print(f"   Transition state [{ind_TS}]")
    print(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(e_zp_TS, 5)).replace('.',',').ljust(12)} eV")
    print(f"Activation energy:\n{'  Forward:'.ljust(indent)} {str(round(Ea_f, 5)).replace('.',',').ljust(12)} eV\n{'  Reverse:'.ljust(indent)} {str(round(Ea_r, 5)).replace('.',',').ljust(12)} eV")
    print(f"{'  dE:'.ljust(indent)} {str(round(dE, 5)).replace('.',',').ljust(12)} eV\n")
    # Final structure
    print(f"   Final stucture")
    print(f"{'Extrapolated ZP:'.ljust(indent)} {str(round(e_zp_final, 5)).replace('.',',').ljust(12)} eV")
    print(f"{'='*100}\n")

    if open_gui:     
        from ase.visualize import view
        view(TS)


def write_poscar(atoms, directory='./') -> None:
    from ase.calculators.vasp import Vasp
    from ase.io.vasp import write_vasp as ase_write_vasp

    atoms.calc = calc = Vasp(setups=vasp_pot_collections['VASP'], xc='PBE')

    atoms.calc.initialize(atoms)
    ase_write_vasp(os.path.join(directory, "POSCAR"),
               calc.atoms_sorted,
               symbol_count=calc.symbol_count,
               ignore_constraints=calc.input_params['ignore_constraints'])


def auto_convert(filepath: str=None, targetext: str='traj', args=None) -> None:
    from ase.io import read, write

    if args:
        filepath = args.file

    filepath = os.path.abspath(filepath)
    directory, filename = os.path.split(filepath)
    if not filepath.__contains__("@"):
        filepath += "@:"
    atoms = read(filepath)
    _, ext = os.path.splitext(filename)
    
    if filename in {'POSCAR', 'CONTCAR', 'OUTCAR', 'XDATCAR'}:
        target = f"coordinates.{targetext}"
        target = os.path.join(directory, target)
        hasPrevious = os.path.exists(target)
        write(target, atoms, append=hasPrevious)
    elif ext in {'.xyz', '.traj'}:
        write_poscar(atoms=atoms[-1], directory=directory)



if __name__ == "__main__":
    arpar = argparse.ArgumentParser(description='Useful tools for dealing with DFT input / output')
    parsers = arpar.add_subparsers(description='Valid commands')
    arpar.add_argument("file", type=str, help="File")
    
    par_show = parsers.add_parser('show')
    par_show.add_argument('-o', action="store_true", help="Open in GUI")
    par_show.set_defaults(function=show_results)

    par_neb = parsers.add_parser('neb')
    par_neb.add_argument("nimages", type=str, help="The number of images to read from the end")
    par_neb.add_argument("-o", action="store_true", help="Open the transition state in ASE")
    par_neb.set_defaults(function=nebtool)

    par_pot = parsers.add_parser('pot')
    par_pot.add_argument("-p", type=str, default='GW', help="Select set of PAWs to use. Currently 'VASP', 'MIN' or 'GW'. Default: 'GW'")
    par_pot.set_defaults(function=write_potcar)

    par_convert = parsers.add_parser('convert')
    par_convert.set_defaults(function=auto_convert)
    
    args = arpar.parse_args()

    tool = args.function
    tool(args=args)
    
