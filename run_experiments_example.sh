#!/bin/bash
#./run_experiments.sh


python CD_main.py --alpha 0.05 --taumax 5 --vars 'AMOC_era5','CLLJ','SAR_prec','NDVI' --varnames 'AMOC','CLLJ','PREC','NDVI' --CI ParCorr --mask 'Southern AR dry season' --show-table
