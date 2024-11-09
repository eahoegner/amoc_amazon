# amoc_amazon

code for the AMOC --> Southern Amazon Rainforest tipping interaction paper

indices-causal-stationarity.ipynb  to construct the indices for the causal stationarity analysis; output stored in ERA5_robustness-allvars.csv
indices-ERA5.ipynb  to construct indices for causal analysis with ERA5 precipitation index; output stored in ERA5_1982-2022.csv
indices-GPCC.ipynb  to construct indices for causal analysis with GPCC precipitation index; output stored in GPCC_1982-2020.csv
indices-robustness.ipynb  to construct and collect indices for additional robustness; output stored in ERA5_robustness-allvars.csv

additional input data is needed to run the indices- notebooks; let me know in case you want to reproduce them; however, you can see what happened here, and the output .csv files are available in the data folder

CE-bootstrapping.ipynb  notebook to run the causal effect bootstrapping analysis with, creates parts of Figure 3
causal-map-analysis.ipynb  notebook to run the causal map analysis with; requires additional input data; creates Figure 4, Suppl. Figures 3-7
causal-stationarity.ipynb  notebook to run causal stationarity analysis, creates all components of Figure 5

CD_main.py   run causal discovery; requires one of the .csv files as input; used to produce all causal graphs, f.ex. for Figure 2, 3
run_experiments_example  example input for running CD_main.py
