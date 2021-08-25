# Methods

This subpackage contains methos that can be used for habitat modelling. When packages have progressed from the experimentation stage and are ready to be used in deployment situations, they should be moved here.

The currently available methods are:
- feature\_extraction - Performs feature extraction on data, including rasters and images. Primarily focussed on autoencoders, but applicable to any model.
- cluster - performs clustering for habitat modelling. Doesn't implement clustering algorithms, as these should be sourced from external packages or 'swim'.
- latent\_mapping - peforms habitat mapping by mapping between a latent space and given targets.


