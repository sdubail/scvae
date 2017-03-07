#!/usr/bin/env python3

import data
import modeling
import analysing

import os
import argparse

def main(data_set_name, data_directory, log_directory, results_directory,
    splitting_method, splitting_fraction,
    latent_size, hidden_sizes, reconstruction_distribution,
    number_of_reconstruction_classes,
    number_of_warm_up_epochs, batch_normalisation, count_sum,
    number_of_epochs, batch_size, learning_rate,
    reset_training):
    
    log_directory = os.path.join(log_directory, data_set_name)
    results_directory = os.path.join(results_directory, data_set_name)
    
    # Data
    
    data_set = data.DataSet(data_set_name, data_directory)
    
    print()
    
    training_set, validation_set, test_set = data_set.split(
        splitting_method, splitting_fraction)

    feature_size = data_set.number_of_features
    
    print()
    
    # Modeling

    model = modeling.VariationalAutoEncoder(
        feature_size, latent_size, hidden_sizes,
        reconstruction_distribution,
        number_of_reconstruction_classes,
        batch_normalisation, count_sum, number_of_warm_up_epochs,
        log_directory = log_directory
    )
    
    print()
    
    model.train(training_set, validation_set,
        number_of_epochs, batch_size, learning_rate,
        reset_training)
    
    print()
    
    reconstructed_test_set, latent_set, test_metrics = model.evaluate(
        test_set, batch_size)
    
    # Analysis
    
    analysis = analysing.Analysis(
        log_directory, results_directory, model.name
    )
    
    analysis.analyseModel()

    analysis.analyseResults(test_set, reconstructed_test_set, latent_set)

parser = argparse.ArgumentParser(
    description='Model single-cell transcript counts using deep learning.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--data-set-name",
    type = str,
    default = "sample",
    help = "name of data set"
)
parser.add_argument(
    "--data-directory",
    type = str,
    default = "data",
    help = "directory where data is placed"
)
parser.add_argument(
    "--log-directory",
    type = str,
    default = "log",
    help = "directory where models are logged"
)
parser.add_argument(
    "--results-directory",
    type = str,
    default = "results",
    help = "directory where results are saved"
)
parser.add_argument(
    "--splitting-method",
    type = str,
    default = "random", 
    help = "method for splitting data into training, validation, and test sets"
)
parser.add_argument(
    "--splitting-fraction",
    type = float,
    default = 0.8,
    help = "fraction to use when splitting data into training, validation, and test sets"
)
parser.add_argument(
    "--latent-size",
    type = int,
    default = 50,
    help = "size of latent space"
)
parser.add_argument(
    "--hidden-sizes",
    type = int,
    nargs = '+',
    default = [500],
    help = "sizes of hidden layers"
)
parser.add_argument(
    "--reconstruction-distribution",
    type = str,
    default = "poisson",
    help = "distribution for the reconstructions"
)
parser.add_argument(
    "--number-of-reconstruction-classes",
    type = int,
    default = 0,
    help = "the maximum count for which to use classification"
)
parser.add_argument(
    "--number-of-epochs",
    type = int,
    default = 100,
    help = "number of epochs for which to train"
)
parser.add_argument(
    "--batch-size",
    type = int,
    default = 100,
    help = "batch size used when training"
)
parser.add_argument(
    "--learning-rate",
    type = float,
    default = 1e-3,
    help = "learning rate when training"
)
parser.add_argument(
    "--number-of-warm-up-epochs",
    type = int,
    default = 0,
    help = "number of epochs with a linear weight on the KL-term"
)
parser.add_argument(
    "--batch-normalisation",
    action = "store_true",
    help = "use batch normalisation"
)
parser.add_argument(
    "--no-batch-normalisation",
    dest = "batch_normalisation",
    action = "store_false",
    help = "do not use batch normalisation"
)
parser.set_defaults(batch_normalisation = True)
parser.add_argument(
    "--count-sum",
    action = "store_true",
    help = "use count sum"
)
parser.add_argument(
    "--no-count-sum",
    dest = "count_sum",
    action = "store_false",
    help = "do not use count sum"
)
parser.set_defaults(count_sum = True)
parser.add_argument(
    "--reset-training",
    action = "store_true",
    help = "reset already trained model"
)

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
