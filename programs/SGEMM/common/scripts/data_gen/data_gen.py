import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Data Generator',
        description=
            'This program produces a input of size MxN\n\n',
        usage=f'{sys.argv[0]} '
        'M N '
        '--output-file OUTPUT_URL'
    )

    parser.add_argument(
        'M',
        type=int,
        help='number of rows for the output'
    )

    parser.add_argument(
        'N',
        type=int,
        help='number of columns for the output'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='output file\'s url'
    )

    return parser

def main():
    parser = setup_args()
    args = parser.parse_args()

    m = args.M
    n = args.N
    output_url = args.output_file

    with open(output_url, "w") as output_file :

        output_file.write(f"{m} {n}\n")
        for _ in range(m) :

            rng = np.random.default_rng()
            random_row = rng.uniform(-100, 101, size=n)

            for i in random_row :
                output_file.write(f"{i:.2f} ")

            output_file.write("\n")

    return 0

if __name__ == '__main__' :
    main()