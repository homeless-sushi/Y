import argparse
import os
import sys

PROGRAM_URL = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../ALGORITHM/build/CUTCP/CUTCP")) # this script directory
 
def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Run Precision',
        description=
            'This program runs the CUTCP algorithm at varying levels of precision'
            'and plots the results.\n\n',
        usage=f'{sys.argv[0]} '
        '--input-file INPUT_FILE '
        '--output-dir OUTPUT_DIR'
        'max_p min_p'
    )

    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='input file\'s url'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='output directory\'s url'
    )

    parser.add_argument(
        'max_p',
        type=int,
        help='max precision included'
    )

    parser.add_argument(
        'min_p',
        type=int,
        help='min precision included'
    )

    return parser

def main() :
    parser = setup_args()
    args = parser.parse_args()

    input_url = args.input_file
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok = True) 

    max_p = args.max_p
    if max_p > 100 :
        raise ValueError("ERROR: ILLEGAL ARGUMENT - max precision should be <= 100")
    min_p = args.min_p
    if min_p < 0 :
        raise ValueError("ERROR: ILLEGAL ARGUMENT - min precision should be >= 0")
    if max_p < min_p :
        raise ValueError("ERROR: ILLEGAL ARGUMENT - max precision should be greater than or equal to min precision")


    for precision in range(max_p, min_p-1, -1) :
        precision = str(precision)
        
        output_url = os.path.join(out_dir, precision + ".txt")

        if os.path.isfile(output_url) :
            continue

        os.system(' '.join([
                PROGRAM_URL,
                '-I',input_url,
                '-O',output_url,
                '--precision',precision
        ]))

    return

if __name__ == '__main__' :
    main()