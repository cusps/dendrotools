import sys
import getopt
import os

from .full_image import FullImage

help = """"""

def main(argv):
    # if "-d" in args:
    #     set_defaults(args)
    opts = []
    params = []
    try:
        opts, params = getopt.getopt(argv, "hf:o:l:", ["file=", "output=", "folder="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
    file = ''
    output = ''
    folder = ''
    for opt, arg in opts:
        if opt == 'h':
            print(help)
            exit()
        elif opt in ('-f', '--file'):
            file = arg
        elif opt in ('-o', '--output'):
            output = arg
        elif opt in ('-l', '--folder'):
            folder = arg

    if output:
        if file:
            _run_full_image(file, output)
        elif folder:
            _run_folder(file, output)
        else:
            raise ValueError("Need to specify input location.")
    else:
        raise ValueError("Need to specify output location.")


def _run_full_image(input, output):
    if '.tiff' in input:
        FullImage(input, output)


def _run_folder(input, output):
    for f in os.listdir(input):
        _run_full_image(f, output)


if __name__ == "__main__":
    main(sys.argv[1:])
