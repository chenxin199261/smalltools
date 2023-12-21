import numpy as np
from tool_union_find import *
#++++++++++++++++++++++
#
#  Dictionaries
#
#++++++++++++++++++++++

radii_dict = { 
        'H': 0.31, 'He': 0.28,
        'Li': 1.21, 'Be': 0.96, 'B': 0.84, 'C': 0.69, 'N': 0.71, 
        'O': 0.66, 'F': 0.64, 'Ne': 0.58,
        'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07,
        'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
        'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53,
        'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24,
        'Cu': 1.32, 'Zn': 1.22                                                                 
}                                                                          

#==============================
def BuildMaskForXYZ(Element):
        # Initialize global Mask Matrix
        MaskMat = np.array([*map(radii_dict.get, Element)],dtype=np.float16)
        MaskMat = np.tile(MaskMat,(len(Element),1))
        MaskMat = (MaskMat + MaskMat.T)*1.4
        np.fill_diagonal(MaskMat,0)
        return MaskMat

def max_distance(set1, set2):
    from scipy.spatial.distance import cdist
    # Use cdist to calculate pairwise distances between points in set1 and set2
    #chatGPT
    distances = cdist(set1, set2)

    # Find the indices of the maximum distance
    max_distance_index = np.unravel_index(np.argmax(distances), distances.shape)

    # Extract the points corresponding to the maximum distance
    point1 = set1[max_distance_index[0]]
    point2 = set2[max_distance_index[1]]

    return max_distance_index, point1, point2, np.max(distances)


def frag_unwarp(atoms):

    from ase import Atoms
    # This function partition system into group
    # atoms is ASE atoms class
    distMat = atoms.get_all_distances(mic=True) 
    elements = atoms.get_chemical_symbols()
    mask = BuildMaskForXYZ(elements) 
    cell = atoms.get_cell() 
    # Mask for fragment
    LinkMat = distMat < mask

    for i in range(len(elements)):
        LinkMat[i][i] = False


    # Split atoms into submolecules
    position_list = atoms.get_positions()
    uf = groupSplit(LinkMat)
    MolRec = uf.components()
    nMol = len(MolRec)
   

    # Check if it cross boundary.
    Molecules = []
    for imol in MolRec:
        Elements = []
        Coords   = []

        for iatm in imol:
            Elements.append(elements[iatm])
            Coords.append(position_list[iatm])
        subMol = Atoms(Elements,Coords,cell=cell,pbc=True)
        # Check is it cross boundary.
        mask = BuildMaskForXYZ(Elements) 
        distMat = subMol.get_all_distances(mic=False) 
        LinkMat = distMat < mask
        for i in range(len(Elements)):
            LinkMat[i][i] = False

        uf = groupSplit(LinkMat)
        MolRec = uf.components()
        nMol = len(MolRec) # nMol >1 means crossboundary.
        if (nMol > 1):
            coord_new = []
            # Define the box vectors (non-orthogonal)
            box_vectors = np.array(cell)
            # Define the coordinates of points A and B
            #coord_new.append(list(Coords[0]))
            coord_new = Coords
           
            imaxsub = None
            maxsub = 0
            for isub in MolRec:
                if(len(isub)>maxsub):
                    imaxsub = list(isub)
                    maxsub = len(isub)
            Coords_max = np.array(Coords)[imaxsub]

            
            for isub in MolRec:
                if (imaxsub[0] in isub): continue
                Coords_move = np.array(Coords)[list(isub)]
                # Compute largest distance between 2 fragment.
                idx,point_A,point_B,maxdist = max_distance(Coords_max, Coords_move) 

                distance = point_B - point_A
                wrapped_distance = np.linalg.solve(box_vectors.T, distance.T).T
                wrapped_distance = np.round(wrapped_distance)

                # The move should based on the largest inter-atomic distanse
                move = np.dot(wrapped_distance, box_vectors) 

                for iatom in isub:
                    coord_new[iatom] = coord_new[iatom] - move 
            # Change the coordinate of global 
            i = 0
            for iatm in imol:
                position_list[iatm] = coord_new[i] 
                i = i+1
        else:
            coord_new = Coords

        for icoord in range(len(coord_new)):
            coord_new[icoord] = list(coord_new[icoord])

        Molecules.append([Elements,coord_new])

    atoms.set_positions(position_list)
    return atoms, Molecules


