import argparse
import yaml
import sys
import logging

import pandas as pd
from pathlib import Path
from datetime import datetime
from muvis.train.run_nn_experiments import run_experiment as run_nn
from muvis.train.run_tree_experiments import run_experiment as run_tree


def run_single_experiment(config_path, log_level="INFO"):
    """Run a single experiment."""
    print(f"Reading config from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    
    exp_type = conf["experiment_type"]

    if exp_type == "nn":
        run_nn(conf, log_level=log_level)
    elif exp_type == "tree":
        run_tree(conf, log_level=log_level)
    else:
        print(f"Error: Unknown experiment type '{exp_type}' in config.")
        sys.exit(1)

def run_multiple_experiments(config_paths, output_file=None, log_level='INFO'):
    """Run multiple experiments and collect results directly into a wide format."""
    wide_results = {}
    
    for config_path in config_paths:
        print(f"{'='*60}")
        print(f"Running: {config_path}")
        print(f"{'='*60}")
        
        with open(config_path, 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        
        exp_type = conf["experiment_type"]
        
        if exp_type == "nn":
            model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper = run_nn(conf, log_level=log_level)
        elif exp_type == "tree":
            model, val_rmse, test_rmse, boot_mean, ci_lower, ci_upper = run_tree(conf, log_level=log_level)
        else:
            raise ValueError(f"Unknown experiment type '{exp_type}'")
        
        model_name = Path(config_path).stem
        dataset_name = Path(config_path).parent.name
        
        print(f"✓ {model_name}: Test RMSE = {test_rmse:.4f}")
        
        if dataset_name not in wide_results:
            wide_results[dataset_name] = {}
            
        wide_results[dataset_name].update({
            f"{model_name}_test_rmse": test_rmse,
            f"{model_name}_boot_mean": boot_mean,
            f"{model_name}_ci_lower": ci_lower,
            f"{model_name}_ci_upper": ci_upper
        })
    
    wide_df = pd.DataFrame.from_dict(wide_results, orient='index')
    wide_df.index.name = 'dataset'
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiment_results_{timestamp}.csv"
    
    wide_df.to_csv(output_file)

    return wide_df


def main():
    parser = argparse.ArgumentParser(description='Run Experiments')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    single_parser = subparsers.add_parser('single', help='Run a single experiment')
    single_parser.add_argument('--runconf', type=str, required=True, 
                              help='Path to the YAML config file')
    
    batch_parser = subparsers.add_parser('batch', help='Run multiple experiments')
    batch_parser.add_argument('--configs', nargs='+', required=True,
                             help='Paths to YAML config files')
    batch_parser.add_argument('--output', type=str, default=None,
                             help='Output CSV file path (default: timestamped)')
    
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'None'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    if args.log_level == "None":
        args.log_level = None
    
    if args.command == 'single':
        run_single_experiment(args.runconf, log_level=args.log_level)
    elif args.command == 'batch':
        run_multiple_experiments(args.configs, args.output, log_level=args.log_level)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()