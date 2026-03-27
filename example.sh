#!/bin/bash
#SBATCH --job-name RedMapperRoman

python -u -m RedmapperRoman.Runner \
		--outdir /scratch/midway3/dhayaa/Roman/Cardinal_20260308/ \
		--bands F106,F129,F158,F184 --bands_inds 0,1,2,3 --survey roman \
		--input_catalog_hdf5 /project/chihway/dhayaa/Roman/Cardinal/MockRun/RomanGold.hdf5 \
		--input_specz /project/chihway/dhayaa/Roman/Cardinal/MockRun/SpeczSample.fits \
		--fracdet_map /project/chihway/dhayaa/Roman/Cardinal/MockRun/Masks/footprint_mask.hpy \
		--foreground_map /project/chihway/dhayaa/Roman/Cardinal/MockRun/Masks/foreground_mask.hpy \
		--SPmap_path /project/chihway/dhayaa/Roman/Cardinal/MockRun/SPmaps/MAP \
		--n_jobs 28 --refband F184 \
		--color_presel_thresh 0.2,0.2,0.2,0.2 \
		--z_range 0.1,0.95

