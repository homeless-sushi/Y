import argparse
import random
import struct
import sys

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Data Generator',
        description=
            'This program produces a input of height Y and width X\n\n',
        usage=f'{sys.argv[0]} '
        'X Y '
        '--output-file OUTPUT_URL'
    )

    parser.add_argument(
        'x',
        type=int,
        help='width of output'
    )

    parser.add_argument(
        'y',
        type=int,
        help='height of output'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='output file\'s url'
    )

    return parser

def generate_random_numbers(X):
    return [(random.randint(0, 255)) for _ in range(3*X)]

def generate_random_line_x(X):
    return ' '.join(str(random.randint(0, 255)) for _ in range(3*X))

def generate_y_random_lines(X, Y):
    return '\n'.join(generate_random_line_x(X) for _ in range(Y))

def main():
    parser = setup_args()
    args = parser.parse_args()

    width = args.x
    height = args.y
    output_url = args.output_file

    with open(output_url, "wb") as output_file :
        output_file.write(struct.pack('II', width, height))

        # Generate and write pixel data
        for _ in range(height):
            pixel_data = generate_random_numbers(width)
            output_file.write(struct.pack(f'{width*3}H', *pixel_data))

    return 0

if __name__ == '__main__' :
    main()