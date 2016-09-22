# ASAC
Shared code base for Amy and Adam

## File Structure for RAM data
.
|---VISRAM
|
└-------Mouse 1,2,3, etc.
		|
		└-------Day 1,2,3, etc.
			|
			└-------Behavior
			|
			└-------Calcium
			|	|
			|	└-------Raw
			|	|
			|	└-------Decompressed
			|	|
			|	└-------Movies
			|	|
			|	└------- CellMax (for rec_ files, class.txt, output
			|
			└-------Analysis (for cell traces and behavior tracking info).


## File naming convention
### Processing Movies
movie processing steps will be saved as hdf5 files, with the .h5 extension. They will be named according to the mouse, recording area, day, and the processing steps, as follows:
m1_visp_d1_ds_nhp.h5 (after loading, mouse 1, primary visual cortex, day 1, downsampled, no hot pixels)
m1_visp_d1_reg.h5 (after motion registration, mouse 1, primary visual cortex, day 1, registered)
m1_visp_d1_proc.h5 (after processing, this goes into cellmax, mouse 1, primary visual cortex, day 1, processed))

## Basic processing steps
### Load Files, Spatially Downsample, Median Filter
### Motion Registration
### Temporal Downsample
### Processing
