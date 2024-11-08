# scVAE: Single-cell variational auto-encoders #

scVAE is a command-line tool for modelling single-cell transcript counts using variational auto-encoders.

### For developers (on this forked version)

Dependencies were pinned :

Install the scVAE using python 3.7. For Apple Silicon users set your conda environment to osx-64: 

	$ CONDA_SUBDIR=osx-64 conda create -n myenv python=3.7
	conda activate myenv
	conda config --env --set subdir osx-64

Then : 

	$ python setup.py install

### For users of the original package:  
Install scVAE using pip for Python 3.6 and 3.7:

	$ python3 -m pip install scvae

scVAE can then be used to train a variational auto-encoder on a data set of single-cell transcript counts:

	$ scvae train transcript_counts.tsv

And the resulting model can be evaluated on the same data set:

	$ scvae evaluate transcript_counts.tsv

For more details, see the [documentation][], which include a user guide and a short tutorial.

[documentation]: https://scvae.readthedocs.io
