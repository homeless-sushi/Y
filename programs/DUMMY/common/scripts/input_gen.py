import argparse
import random
import sys

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Random Input Generator',
        description=
            'This program generates a random input file with n values.\n'
            '\n',
        usage=f'{sys.argv[0]} n --min --max output-file'
    )

    parser.add_argument(
        'n',
        type=int,
        help='the number of datapoints'
    )

    parser.add_argument(
        '--min',
        type=float,
        help='the min value'
    )

    parser.add_argument(
        '--max',
        type=float,
        help='the max value'
    )

    parser.add_argument(
        'output-file',
        type=str,
        help='the url of the output file'
    )

    return parser


def main() :
    parser = setup_args()
    args = parser.parse_args()

    n = args.n
    min = 0 if args.min == None else args.min
    max = 1 if args.max == None else args.max


    out_url = getattr(args, "output-file")
    with open(out_url, "w") as out_file :
        for _ in range(n):
            random_value = random.uniform(min, max)
            out_file.write(f"{random_value}\n")

    
if __name__ == "__main__" :
    main()