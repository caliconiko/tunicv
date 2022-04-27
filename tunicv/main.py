import argparse
from pathlib import Path

from tunicv.reader import Reader

def main():
    arg_parser = argparse.ArgumentParser(
        description="Read trunes from images"
    )

    arg_parser.add_argument(
        "path",
        type=str,
        help="Path to the image"
    )

    args = arg_parser.parse_args()

    path = Path(args.path)

    if not path.is_file():
        print("File doesn't exist")
    else:
        reader = Reader(path)
        reader.read()