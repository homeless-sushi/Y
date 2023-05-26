import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__)) # this script directory
BUILD_DIR = "build" # the build directory
PROGRAMS_DIR = "programs" # the programs' sources directory
DATA_DIR = "data" # the programs' data directory
INPUT_DIR = "in" # the programs' input data directory
OUTPUT_DIR = "out" # the programs' output data directory

def get_program_url(name, mode) :
    return os.path.join(
        PROJECT_DIR,
        PROGRAMS_DIR,
        name,
        mode,
        BUILD_DIR, 
        name, 
        name
    )

def get_input_dir(name) :
    return os.path.join(
            PROJECT_DIR,
            PROGRAMS_DIR, 
            name,
            DATA_DIR,
            INPUT_DIR
        )

def get_input_url(name, input_file) : 
    return os.path.join(
            get_input_dir(name),
            input_file
        )

def get_output_dir(name) :
    return os.path.join(
            PROJECT_DIR,
            PROGRAMS_DIR, 
            name,
            DATA_DIR,
            OUTPUT_DIR
        )
    
def get_output_url(name, output_file) : 
    return os.path.join(
            get_output_dir(name),
            output_file
        )

def add_name_argument(parser) :
    parser.add_argument(
        'name',
        type=str,
        help='the name of the benchmark'
    )

def add_mode_argument(parser, required=True) :
    parser.add_argument(
        'mode',
        type=str,
        nargs= None if required else '?',
        help=
        'Must be ALGORITHM | PROFILING | BENCHMARK: '
        'ALGORITHM: tests the algorithm; '
        'PROFILING: profiles the application with agora; '
        'BENCHMARK: benchmarks the application connecting to the controller;'
    )
   
def run(args) :

    parser = argparse.ArgumentParser(
        prog='Run',
        description=
            'Option RUN runs the algorithms, profiling applications, '
            'and benchmarks of this suite.\n\n',
        usage=f'{sys.argv[0]} RUN '
        'name '
        'ALGORITHM | PROFILING | BENCHMARK '
        '--input-file '
        '--output-file '
        '[other options] '
    )

    add_name_argument(parser)
    add_mode_argument(parser, True)

    parser.add_argument(
        '-I','--input-file',
        type=str,
        required=True,
        help='name of the input data set'
    )

    parser.add_argument(
        '-O','--output-file',
        type=str,
        required=True,
        help='name of the output file'
    )

    args, other_options = parser.parse_known_args(args)
    program_url = get_program_url(args.name.upper(), args.mode.upper())
    input_url = get_input_url(args.name.upper(), args.input_file)
    output_url = get_output_url(args.name.upper(), args.output_file)

    os.system(' '.join([
            program_url,
            '-I',input_url,
            '-O',output_url,
            ' '.join(other_options)
    ]))

def help(args) :

    parser = argparse.ArgumentParser(
        prog='Help',
        description=
            'Option HELP shows information about the programs of this suite, '
            'their inputs, and their options.\n\n',
        usage=
        f'{sys.argv[0]} HELP \n: shows available programs\n'
        f'{sys.argv[0]} HELP name --inputs: shows available inputs\n'
        f'{sys.argv[0]} HELP name mode: shows program\'s options\n'
    )

    add_name_argument(parser)
    add_mode_argument(parser, False)

    parser.add_argument(
        '--inputs',
        action='store_true'
    )
    parser.set_defaults(i=False)

    args = parser.parse_args(args)

    if args.inputs :
        input_dir = get_input_dir(args.name.upper())
        files = os.listdir(input_dir)
        files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]
        print('Available inputs are:\n' + ' '.join(files))
        return
    
    program_url = get_program_url(args.name.upper(), args.mode.upper())
    os.system(' '.join([
        program_url,
        '--help'
    ]))

def main() :

    parser = argparse.ArgumentParser(
        prog='Run',
        description=
            'Option RUN runs the algorithms, profiling applications,\n'
            'and benchmarks of this benchmark suite.\n\n',
        usage=f'{sys.argv[0]} '
        'RUN | HELP '
        
    )

    parser.add_argument(
        'action',
        type=str,
        help=
        'Must be RUN | HELP: '
        'RUN: runs programs of this suite; '
        'HELP: shows information about the programs of this suite;'
    )

    args, other_options = parser.parse_known_args()
        
    actions = {
        'RUN' : run,
        'HELP' : help
    }
    actions[args.action.upper()](other_options)

    return

if __name__ == '__main__' :
    main()