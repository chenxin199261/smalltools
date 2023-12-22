import ase
from ase.io import read, write
from ase import Atoms
import numpy as np
from tool_fragmentation import frag_unwarp 


# Tools
def minimum_image_distance(r1, r2, box_vectors):
    """
    Calculate the minimum image distance between two points.
    ChatGPT gen.
    """
    
    # r1 and r2 are the positions of the two points
    # box_vectors is a 3x3 matrix representing the box vectors

    # Calculate fractional coordinates
    f1 = np.linalg.solve(box_vectors.T, r1)
    f2 = np.linalg.solve(box_vectors.T, r2)

    # Find integer multiples that minimize the distance
    n = np.round(f2 - f1)

    # Minimum image distance
    d_min = r2 - r1 - np.dot(n, box_vectors)
    return np.linalg.norm(d_min)


def atomic_distance_mat(atom_coords, box_vectors):
  """
  In  : atom_coords (n_atoms, 3)
        box_vectors (3, 3)
  Out : distance_mat (n_atoms, n_atoms)
  Calculate the distance matrix for a set of atoms.
  """
  n_atoms = len(atom_coords)
  distance_mat = np.zeros((n_atoms, n_atoms))
  for i in range(n_atoms):
    for j in range(i+1, n_atoms):
      distance_mat[i, j] = minimum_image_distance(atom_coords[i], atom_coords[j], box_vectors)
      distance_mat[j, i] = distance_mat[i, j]
  return distance_mat



def main_cif2xyz(directory, fileName):
    structure = ase.io.read(directory+fileName)

    # Get coord and cell dimension.
    xyz_coords = structure.get_positions()
    cell = structure.get_cell()

    
    #get distance matrix
    structure_new,mols = frag_unwarp(structure)
    i = 0
    for imol in mols:
        subMol = Atoms(imol[0],imol[1],cell=cell,pbc=True)
        ase.io.write(directory+'mol'+str(i)+'.xyz', subMol)
        i=i+1


if __name__ == '__main__':
    #main_cif2xyz('./example/','CL20-TNT.cif')
    main_cif2xyz('./example/','15.cif')
