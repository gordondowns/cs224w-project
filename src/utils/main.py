# import sys
# sys.path.append("..")
# from src.data import load_crystal_structure
# from src.utils import load_crystal_structure
# from . import load_crystal_structure
from load_crystal_structure import load_single_crystal_structure

if __name__ == "__main__":
    # fp = "data/raw/CIFs/Durangite0019593.cif"
    fp = "../../data/raw/CIFs/Durangite0019593.cif"
    # cs = load_crystal_structure.load_single_crystal_structure(fp)
    cs = load_single_crystal_structure(fp)
    # print(cs)
