import argparse
import os
import sys

import pandas as pd

def read_data(url) :
    df = pd.read_csv(
        url,
        skipinitialspace = True,
        comment="#"
    )

    return df

def absolute_error(reference, approximate) :

    diff = reference - approximate
    sqrDiff = diff.applymap(lambda x: x**2)
    error = sqrDiff.values.sum()

    return error

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Compute Error',
        description=
            'This program reads the results of NBODY algorithm'
            'and computes absolute and relative errors.\n'
            'Results should be named PRECISION.txt,'
            'where precision is an int\n\n',
        usage=f'{sys.argv[0]} '
        '--max-precision MAX_PRECISION '
        '--min-precision MIN_PRECISION '
        '--results-dir RESULTS_DIR '
        '--output-file OUT_FILE'
    )

    parser.add_argument(
        '--max-precision',
        type=str,
        required=True,
        help='url of the output with max precision'
    )

    parser.add_argument(
        '--min-precision',
        type=str,
        required=True,
        help='url of the output with min precision'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='url of the dir with results with various precision'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='output file\'s url'
    )

    return parser

def main() :
    parser = setup_args()
    args = parser.parse_args()

    files = os.listdir(args.results_dir)
    files = [f for f in files if os.path.isfile(
        os.path.join(args.results_dir,f)
    )]

    with open(args.output_file, "w") as result_file :

        result_file.write("PRECISION,ABSOLUTE_ERR,RELATIVE_ERR\n")

        max_precision = read_data(args.max_precision)
        min_precision = read_data(args.min_precision)
        max_error = absolute_error(max_precision, min_precision)

        for file in files :
            precision = int(os.path.splitext(file)[0])

            approximate_data = read_data(os.path.join(args.results_dir,file))
            absolute_err = absolute_error(max_precision, approximate_data)

            relative_err = absolute_err/max_error

            result_file.write(f"{precision},{absolute_err},{relative_err}\n")

    return

if __name__ == '__main__' :
    main()