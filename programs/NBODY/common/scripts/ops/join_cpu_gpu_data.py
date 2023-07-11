import argparse
import re
import sys

def setup_args() :

    parser = argparse.ArgumentParser(
        prog='Join CPU GPU data',
        description=
            'This program completes CPU and GPU csv, and joins them\n\n',
        usage=f'{sys.argv[0]} cpu-file gpu-file output-file'
    )

    parser.add_argument(
        'cpu-file',
        type=str,
        help='name of the cpu file'
    )

    parser.add_argument(
        'gpu-file',
        type=str,
        help='name of the gpu file'
    )

    parser.add_argument(
        'output-file',
        type=str,
        help='name of the output file'
    )

    return parser

def cpu(cpu_file, out_file) :

    for line_num, line in enumerate(cpu_file) :

        if line_num == 0 :
            continue

        pattern = r'^(?:[^,]+),(?P<cpu_threads>\d+(?:\.\d+)?),(?P<precision>\d+(?:\.\d+)?),(?P<timing>\d+(?:\.\d+)?,\d+(?:\.\d+)?)$'
        data = re.match(pattern, line).groupdict()

        cpu_device = 0
        gpu_blocks_exps = [0,1,2,3,4,5]
        for gpu_blocks_exp in gpu_blocks_exps :
            out_line = ",".join([
                f"{cpu_device}",
                data["cpu_threads"],
                f"{gpu_blocks_exp}",
                data["precision"],
                data["timing"]
            ])
            out_file.write(out_line + "\n")
    
def gpu(gpu_file, out_file) :
    for line_num, line in enumerate(gpu_file) :

        if line_num == 0 :
            continue
        
        pattern = r'^(?:[^,]+),(?P<gpu_block_exp>\d+(?:\.\d+)?),(?P<precision>\d+(?:\.\d+)?),(?P<timing>\d+(?:\.\d+)?,\d+(?:\.\d+)?)$'
        data = re.match(pattern, line).groupdict()

        gpu_device = 1
        cpu_threads = [1,2,3,4]
        for cpu_thread_num in cpu_threads :
            out_line = ",".join([
                f"{gpu_device}",
                f"{cpu_thread_num}",
                data["gpu_block_exp"],
                data["precision"],
                data["timing"]
            ])
            out_file.write(out_line + "\n")


def main(): 
    parser = setup_args()
    args = parser.parse_args()

    cpu_file_url = getattr(args, "cpu-file")
    gpu_file_url = getattr(args, "gpu-file")
    output_file_url = getattr(args, "output-file")

    with open(cpu_file_url, "r") as cpu_file,\
        open(gpu_file_url, "r") as gpu_file, \
        open(output_file_url, "w") as out_file :

        out_file.write("DEVICE_TYPE,CPU_THREADS,GPU_BLOCK_EXP,PRECISION,timing_avg,timing_std\n")
        cpu(cpu_file, out_file)
        gpu(gpu_file, out_file)

    
if __name__ == '__main__':
    main()