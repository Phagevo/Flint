import os
import subprocess

def prepare(file_path:str) -> str:
    """
    Convert a PDB file to PDBQT format using Open Babel.
    @param file_path: path to the input PDB file
    @return: path to the output PDBQT file
    """

    # Define the output file name
    pdbqt_file = os.path.splitext(file_path)[0] + '.pdbqt'
    pdbqt_file = pdbqt_file.replace('raw', 'preped')
    
    # Convert PDB to PDBQT using Open Babel
    flags = "-xc -xr" if file_path.endswith("pdb") else ""
    command = f'obabel {file_path} -opdbqt -O {pdbqt_file} -h {flags}'
    command += "--partialcharge gasteiger" # include forces and charges
    subprocess.run(command, shell=True)

    return pdbqt_file
