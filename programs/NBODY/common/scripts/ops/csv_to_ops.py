import argparse
import json
import sys

import pandas as pd

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='CSV to OPS',
        description=
            'This program converts profiling data from .csv to an ops.json\n\n',
        usage=f'{sys.argv[0]} input-file output-file'
    )

    parser.add_argument(
        'input-file',
        type=str,
        help='name of the input file'
    )

    parser.add_argument(
        'output-file',
        type=str,
        help='name of the output file'
    )

    return parser


def main() :
    parser = setup_args()
    args = parser.parse_args()

    csv_url = getattr(args, "input-file")
    ops_url = getattr(args, "output-file")

    nbody = []
    df : pd.DataFrame = pd.read_csv(csv_url)
    for _, row in df.iterrows() :
        features = {}
        features["DEVICE_TYPE"] = int(row["DEVICE_TYPE"])
        features["CPU_THREADS"] = int(row["CPU_THREADS"])

        knobs = {}
        knobs["GPU_BLOCK_EXP"] = int(row["GPU_BLOCK_EXP"])
        knobs["PRECISION"] = int(row["PRECISION"])

        metrics = {}
        metrics["timing"] = [row["timing_avg"], row["timing_std"]]

        operating_point = {}
        operating_point["features"] = features
        operating_point["knobs"] = knobs
        operating_point["metrics"] = metrics

        nbody.append(operating_point)

    ops = {}
    ops["nbody"] = nbody

    with open(ops_url, "w+") as ops_file: 
        json.dump(ops, ops_file, indent=8)

    
if __name__ == "__main__" :
    main()