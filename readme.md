fetch_data.py

The original pdb dataset of protein structures is of 50GB. For our pilot purpose of protein structure prediction, we look at a smaller part of this data set using filters which gets us a list of proteins which are sufficiently different from each other to avaoid redundancy.

each protein in pdb dataset has a pdb id. This program extracts pdf ids of 1000 proteins and then downloads the cif files of their structure. 

cif stands for Crystallographic Information File, s a standard, ASCII-based text file format developed by the International Union of Crystallography (IUCr) to store and exchange structural information for crystals, including unit cell dimensions (lattice parameters and angles, space groups, and fractional atomic coordinates)