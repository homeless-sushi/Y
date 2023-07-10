import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REFERENCE_URL = os.path.join(SCRIPT_DIR,"reference","reference.txt")

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Data Generator',
        description=
            'This program produces a input of size N, from a refence file\n\n',
        usage=f'{sys.argv[0]} '
        '--size N '
        '--output-file OUTPUT_URL'
    )

    parser.add_argument(
        '--size',
        type=int,
        required=True,
        help='size of the output'
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

    reference_url = REFERENCE_URL
    n = args.size
    output_url = args.output_file

    # The reference is the Shellwood dataset found here
    # https://bima.astro.umd.edu/nemo/archive/#Sellwood
    #
    # The coordinates are used as is, and not reversed
    # since it's irrelevant.
    #
    # We assume that the particles are randomly generated

    with open(reference_url, "r") as reference_file :

        time_line = next(reference_file)
        data_lines = list(reference_file)[0:n]

        with open(output_url, "w") as output_file :

            output_file.write(time_line)

            for data_line in data_lines :
                
                output_file.write(data_line)

    return 0

if __name__ == '__main__' :
    main()