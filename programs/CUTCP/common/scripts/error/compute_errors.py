import argparse
import os
import sys

def read_data(url) :
    data = []
    with open(url, 'r') as file:
        for line in file:
            try:
                value = float(line.strip())
                data.append(value)
            except ValueError:
                print(f"Invalid line: {line}")
    return data

def absolute_error(reference, approximate) :

    sum_of_squares = 0
    data = list(zip(reference, approximate))
    for tuple in data:
        difference = tuple[1] - tuple[0]
        squared_difference = difference ** 2
        sum_of_squares += squared_difference

    return sum_of_squares

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Compute Error',
        description=
            'This program reads the results of CUTCP algorithm'
            'and computes absolute and relative errors.\n'
            'Results should be named PRECISION.txt,'
            'where precision is an int\n\n',
        usage=f'{sys.argv[0]} '
        '--reference-res REFERENCE_RES '
        '--results-dir RESULTS_DIR '
        '--output-file OUT_FILE'
    )

    parser.add_argument(
        '--reference-res',
        type=str,
        required=True,
        help='url of the output with max precision'
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

        reference_data = read_data(args.reference_res)
        max_error = absolute_error(reference_data, [0] * len(reference_data))

        for file in files :
            precision = int(os.path.splitext(file)[0])

            approximate_data = read_data(os.path.join(args.results_dir,file))
            absolute_err = absolute_error(reference_data, approximate_data)

            relative_err = absolute_err/max_error

            result_file.write(f"{precision},{absolute_err},{relative_err}\n")

    return

if __name__ == '__main__' :
    main()