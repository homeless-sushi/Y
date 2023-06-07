import argparse
import csv
import sys

import matplotlib.pyplot as plt


def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Plot Error',
        description=
            'This program reads the errors of the NBODY algorithm'
            'at various levels of precision and plots them.\n'
            'Results should be named PRECISION.txt,'
            'where precision is an int\n\n',
        usage=f'{sys.argv[0]} '
        '--input-file INPUT_FILE '
        '--output-file OUTPUT_FILE'
    )

    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='input file\'s url'
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

    precision = []
    error = []

    with open(args.input_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        data = [(int(row[0]), float(row[2])) for row in reader]
        data.sort(key=lambda x: x[0])
        
        precision = [row[0] for row in data]
        error = [row[1] for row in data]
        
    plt.plot(precision, error)
    plt.xlabel('Precision')
    plt.ylabel('Relative error')
    plt.title('CUTCP Error')
    plt.savefig(args.output_file)
    plt.show()

if __name__ == '__main__' :
    main()
