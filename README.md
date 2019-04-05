# Ajinomoto Fatty Alcohol Project

This repository implements all of the analytics used for data analysis in this project.  The goal is to provide a readable and reproducible archive. 

![alt text](figures/strain_heatmap.png "Overview of Strains Created in the Project")

## Usage

To Interact with the analysis run the jupyter notebook called [Ajinomoto_Analysis.ipynb](Ajinomoto_Analysis.ipynb). To obtain a complete CSV of all data generated in the project use this link to download the complete csv: [Raw Ajinomoto Data](data/EDD_Ajinomoto_Data.csv).



## Project Overview

The objective of this project was to develop genetically modified *E. Coli* strains which produce the C12 Fatty Alcohol dodecanol at high titer using the design build test learn [DBTL] cycle. This repository contains all of the code used in the learn component of the project to analyze data and recommend strains to make in the next cycle to maximize production. 

In this project we attempted to create 60 strains, 54 were successful.  Those strains encompased 3 different pathway variants, where a pathway variant had different reductase homologs. Dodecanol production varied from 0 to 900 mg/L. Two cycles of the DBTL cycle were performed.  Cycle 2 showed an increase in production of 27% over cycle 1 strains. 

Metabolomics and Proteomics were collected at the final time point of 27 hours. These data were measured with a variety of modalities including, GC-MS, LC-MS, HPLC. The measurement accuracy for each modality is summarized in plots below.

![alt text](figures/GC-MS_error.png)
![](figures/LC-MS_error.png)
![](figures/Targeted_Proteomics_error.png)
![](figures/HPLC_error.png)

*** Copyright Notice ***

"Fatty Alcohol Titer Optimization Analysis v1.0" Copyright (c) 2019, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

