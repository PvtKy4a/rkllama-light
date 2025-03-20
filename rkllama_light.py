import sys
import argparse
from model_run import model_run
from model_list import model_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='rkllama-light')
    parser.add_argument('-list', action='store_true', help='Show a list of available models')
    parser.add_argument('-run', type=str, metavar="<model_name>", help='Run a model from the list of available models')
    args = parser.parse_args()

    if args.list:
        model_list()
    elif args.run:
        model_run(args.run)
    else:
        parser.print_help()

    sys.stdout.flush()
    sys.exit(0)
