# fopt_vengeance
Notebook with script code (and examples based on the paper) for computing zonotopes of feynman matroids

- cone_sign_analysis: To evaluate surviving sign vectors / non-empty cones given constraint matrix
- cone_checker:       To find non-simplicial cones given constraint matrix and basis (optional)
- zonotope_analysis:  To determine (and plot) zonotope given surviving orientations and generators
- dual_analysis:      To determine (and plot) dual zonotope alongside primal zonotope
- full_generic_analysis: Uses above scripts and generalizes it upto 3D
- interactive_3d_plots: same functionality as full_generic_analysis, but the plots are interactive   
- interactive_4d_plots: same as interactive_3d_plots with a heat map for fourth dimension
- pipeline.py         : the MAIN file that streamlines everything into a single pipeline for analysis
- zonotope.ipynb      : notebook with examples and plots from the draft
