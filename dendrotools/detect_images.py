import sys
import getopt
import os

from .full_image import FullImage

HELP = """"""


def main(argv):
    opts = []
    params = []
    try:
        opts, params = getopt.getopt(argv, "hf:o:l:", ["file=", "output=", "folder="])
    except getopt.GetoptError:
        print('test.py [options] <inputfile> <outputfile>')

    for opt, arg in opts:
        if opt == 'h':
            print(HELP)
            exit()

    input = params[0]
    output = params[1]

    detect(input, output)


def detect(input, output):
    if '.tiff' in input:
        _run_full_image(input, output)
    elif os.path.isdir(input):
        _run_folder(input, output)


def _run_full_image(input, output):
    if '.tiff' in input:
        FullImage(input, output)


def _run_folder(input, output):
    for f in os.listdir(input):
        _run_full_image('{}//{}'.format(input, f), output)


if __name__ == "__main__":
    main(sys.argv[1:])
