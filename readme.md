Building a small protein fold predictor

Background:

Proteins are chains of amino acids that spontaneously fold into specific three-dimensional structures. The folded structure of a protein determines how it functions and interacts with other molecules in the cell.

Traditionally, determining the structure of a protein requires experimental techniques such as X-ray crystallography or NMR spectroscopy. These methods are accurate but expensive, time-consuming, and technically challenging, often taking months or even years for a single protein.

Because of this, researchers have long aimed to build computational models that can predict the three-dimensional structure of a protein directly from its amino acid sequence.

Knowing the structure of a protein is extremely valuable. It helps scientists understand biological processes and is particularly important for drug discovery, where drugs are designed to bind to specific protein targets.

A major resource for protein structures is the Protein Data Bank (PDB). It was established in 1971 and stores experimentally determined protein structures. Today, the database contains over 200,000 protein structures and occupies hundreds of gigabytes of structural data.

Recently, researchers at DeepMind developed AlphaFold, a deep learning system capable of predicting protein structures with near-experimental accuracy. This breakthrough significantly advanced the field of structural biology.

For this work, Demis Hassabis was awarded the Nobel Prize in Chemistry in 2024, recognizing the transformative impact of AI on protein structure prediction.


Files

fetch_data.py

The original pdb dataset of protein structures is of 50GB. For our pilot purpose of protein structure prediction, we look at a smaller part of this data set using filters which gets us a list of proteins which are sufficiently different from each other to avaoid redundancy.

each protein in pdb dataset has a pdb id. This program extracts pdf ids of 1000 proteins and then downloads the cif files of their structure. 

cif stands for Crystallographic Information File, s a standard, ASCII-based text file format developed by the International Union of Crystallography (IUCr) to store and exchange structural information for crystals, including unit cell dimensions (lattice parameters and angles, space groups, and fractional atomic coordinates)




steps:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
