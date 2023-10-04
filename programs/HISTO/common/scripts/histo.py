import argparse
import sys

def compute_histo(input_url):
    with open(input_url, 'r') as input_file:
        # Read X and Y from the first line
        X, Y = map(int, input_file.readline().split())

        # Initialize counters for R, G, and B components
        r_counter = [0] * 256
        g_counter = [0] * 256
        b_counter = [0] * 256

        for _ in range(Y):
            # Read the RGB values for each pixel
            pixels = list(map(int, input_file.readline().split()))
            for i in range(0, len(pixels), 3):
                r, g, b = pixels[i], pixels[i+1], pixels[i+2]

                # Increment counters
                r_counter[r] += 1
                g_counter[g] += 1
                b_counter[b] += 1

    return r_counter, g_counter, b_counter


def write_histo(output_url, r, g, b) :
    with open(output_url, 'w') as output_file:
        output_file.write("RED\n")
        for v in r:
            output_file.write(f"{v} ")
        output_file.write("\n")
        output_file.write("GREEN\n")
        for v in g:
            output_file.write(f"{v} ")
        output_file.write("\n")
        output_file.write("BLUE\n")
        for v in b:
            output_file.write(f"{v} ")
        output_file.write("\n")


def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Histo',
        description=
            'This program performs the HISTO algorithm\n\n',
        usage=f'{sys.argv[0]} '
        'INPUT_URL OUTPUT_URL'
    )

    parser.add_argument(
        'input-file',
        type=str,
        help='input file\'s url'
    )

    parser.add_argument(
        'output-file',
        type=str,
        help='output file\'s url'
    )

    return parser


def main():
    parser = setup_args()
    args = parser.parse_args()

    input_url = getattr(args, "input-file")
    output_url = getattr(args, "output-file")

    r,g,b = compute_histo(input_url)
    write_histo(output_url, r,g,b)


if __name__ == '__main__' :
    main()