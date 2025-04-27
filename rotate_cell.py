
from ase.io import read, write
import numpy as np


refax = 2
rotax = 0
aliax = [0., -1., 0.]
inFile = "realign.xyz"



atoms = read(inFile)
projax = atoms.cell[refax].copy()
projax[rotax] = 0.

aliax = np.array(aliax)
alinorm = np.linalg.norm(aliax)
pronorm = np.linalg.norm(projax)

theta = np.arccos(np.vdot(aliax, projax) / (pronorm*alinorm))
direction = np.cross(projax, aliax)[rotax]
direction = direction / np.abs(direction)
theta *= direction


sintheta = np.sin(theta)
costheta = np.cos(theta)

raxis = np.zeros(shape=(3,))
raxis[rotax] = 1.
invaxis = np.nonzero(raxis == 0)[0]


rotatrix = np.array([[costheta, -sintheta],
                     [sintheta,  costheta]])
rotathreex = np.identity(3)
rotathreex[invaxis.reshape(-1,1), invaxis] = rotatrix

from ase.visualize import view
atoms.set_cell(np.transpose(rotathreex @ np.transpose(atoms.cell)))
atoms.set_positions(np.transpose(rotathreex @ np.transpose(atoms.positions)))
view(atoms)