from rdkit import Chem
from Bio.PDB import PDBParser
import numpy as np

def compute_box(
    receptor_path:str, 
    ligand_path:str, 
    cutoff:float=5.0, 
    padding:float=5.0) -> "dict[str, tuple[float, float, float]]":
    
    """
    calculates the dimensions and center of the docking box
    @param receptor_path: path to the receptor file (.pdb)
    @param ligand_path: path to the ligand file (.sdf)
    @param cutoff: capture distance for neighbour atoms (angstrom)
    @param padding: padding around the box to ensure the ligand is inside (angstrom)
    @return: center coordinates (x, y, z) and sizes (x, y, z) of the box
    """
    
    ligand = Chem.SDMolSupplier(ligand_path)[0]
    ligand_coords = np.array([list(ligand.GetConformer().GetAtomPosition(i)) 
        for i in range(ligand.GetNumAtoms())])
    structure = PDBParser(QUIET=True).get_structure('receptor', receptor_path)
    atoms = list(structure.get_atoms()) # get all atoms in receptor

    # compute the geometric center of the ligand (center of mass)
    ligand_center = np.mean(ligand_coords, axis=0).astype(float)
    
    # collect atoms close to the ligand
    site_atoms = np.array([atom.coord 
        for atom in atoms 
        if np.linalg.norm(atom.coord - ligand_center) <= cutoff]).astype(float)
    if site_atoms.size == 0:
        site_atoms = ligand_coords

    # Compute min/max coordinates for the docking box
    x_min, y_min, z_min = np.min(site_atoms, axis=0)
    x_max, y_max, z_max = np.max(site_atoms, axis=0)
    
    return {
        "center": (
            (x_min + x_max) / 2, 
            (y_min + y_max) / 2, 
            (z_min + z_max) / 2
        ),
        "size": (
            (x_max - x_min) + 2 * padding, 
            (y_max - y_min) + 2 * padding, 
            (z_max - z_min) + 2 * padding
        )
    }