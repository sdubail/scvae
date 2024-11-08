# scVAE: Single-cell variational auto-encoders #

scVAE is a command-line tool for modelling single-cell transcript counts using variational auto-encoders.

### For developers (on this forked version)

	The code was migrated from tensorflow 1. to tensorflow 2. for compatibility issues (forever loading imports).

	$ pip install -e your_path_to_project/scvae

### For users of the original package:  
Install scVAE using pip for Python 3.6 and 3.7:

	$ python3 -m pip install scvae

scVAE can then be used to train a variational auto-encoder on a data set of single-cell transcript counts:

	$ scvae train transcript_counts.tsv

And the resulting model can be evaluated on the same data set:

	$ scvae evaluate transcript_counts.tsv

For more details, see the [documentation][], which include a user guide and a short tutorial.

[documentation]: https://scvae.readthedocs.io
