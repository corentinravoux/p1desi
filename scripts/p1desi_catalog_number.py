#!/usr/bin/env python3
import argparse
from p1desi import catalogs

parser = argparse.ArgumentParser(description="p1d catalogs")
parser.add_argument("--name", help="Catalog name", required=True)
parser.add_argument("--type", help="Catalog type", default="QSO", required=False)
parser.add_argument("--extname", help="Special extname", default=None, required=False)

args = vars(parser.parse_args())

catalogs.get_number(args["name"], type=args["type"], extname=args["extname"])
