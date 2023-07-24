import argparse
import os
import math
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REFERENCE_URL = os.path.join(SCRIPT_DIR,"reference","reference.txt")

class Atom:
    def __init__(self, x, y, z, q):
        self.x = x
        self.y = y
        self.z = z
        self.q = q

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
        'X',
        type=float,
        help='len of lattice in X'
    )

    parser.add_argument(
        'Y',
        type=float,
        help='len of lattice in Y'
    )

    parser.add_argument(
        'Z',
        type=float,
        help='len of lattice in Z'
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
    xSize = args.X
    ySize = args.Y
    zSize = args.Z
    output_url = args.output_file

    # The reference is the CUTCP input data included in 
    # the parboil_2.5
    #
    # In the reference particles are not completely random,
    # but since we want to keep the density constant, as it
    # was an assumption in the paper, we can simply truncate
    # the file.

    atoms = []
    with open(reference_url, "r") as reference_file :
        
        for data_line in reference_file:
            x, y, z, q = map(float, data_line.strip().split())
            atoms.append(Atom(x, y, z, q))

        xMin, yMin, zMin = math.inf, math.inf, math.inf

        for atom in atoms:
            xMin = min(xMin, atom.x)
            yMin = min(yMin, atom.y)
            zMin = min(zMin, atom.z)

        xLim = xMin + xSize
        yLim = yMin + ySize
        zLim = zMin + zSize
        atoms = [ atom for atom in atoms if
                    atom.x <= xLim and
                    atom.y <= yLim and
                    atom.z <= zLim ]
        
    with open(output_url, "w") as output_file :
        for atom in atoms :
            output_file.write(f"{atom.x:.3f} {atom.y:.3f} {atom.z:.3f} {atom.q:.3f}\n")

    return 0

if __name__ == '__main__' :
    main()