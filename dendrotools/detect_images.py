import sys
import getopt
import os

from .full_image import FullImage

# TODO: fill in the help
HELP = """usage: detect_images.py [options] -f <file> -o <output> -l <folder> 
options:\n-h\t\thelp\n-f\t\t"""


def main(argv):
    """
    Takes in command line arguments and calls the necessary AI model.

    :param list argv: command line arguments
    """
    opts = []
    params = []
    try:
        opts, params = getopt.getopt(argv, "hf:o:l:", ["file=", "output=", "folder="])
    except getopt.GetoptError:
        print('usage: detect_images.py [options] <inputfile> <outputfile>')

    for opt, arg in opts:
        if opt == 'h':
            print(HELP)
            exit()

    input = params[0]
    output = params[1]

    detect(input, output)


def detect(input, output):
    """Depending on single image or folder input, calls corresponding private function.

    :param string input: tiff image(s) to be detected on with AI model;
        can be a folder or singular image
    :param string output: where results from model should be placed
    """
    if '.tiff' in input:
        _run_full_image(input, output)
    elif os.path.isdir(input):
        _run_folder(input, output)


def _run_full_image(input, output):
    """Run tiff input image through FullImage to get detections through model.

    :param string input: tiff image to be detected on by AI model
    :param string output: folder location where results should be placed
    """
    if '.tiff' in input:
        FullImage(input, output)


def _run_folder(input, output):
    """Iteratively runs model on each tiff image.

    :param string input: folder of tiff images to be detected on by AI model
    :param string output: folder location where results should be placed
    """
    for f in os.listdir(input):
        _run_full_image('{}//{}'.format(input, f), output)


if __name__ == "__main__":
    main(sys.argv[1:])
