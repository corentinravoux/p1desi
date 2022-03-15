#!/usr/bin/env python3
from p1desi import bookkeeping
import argparse

parser = argparse.ArgumentParser(description='p1d bookkeeping')
parser.add_argument('--region', help='Region on which to compute p1d',required=True)
parser.add_argument('--snr-cut', help='SNR cut applied',required=True)
parser.add_argument('--lines-name', help='Sky line file name', default=None,required=False)
parser.add_argument('--catalog-name', help='Quasar catalog name used', default=None,required=False)
parser.add_argument('--dla-name', help='DLA catalog name used', default=None,required=False)
parser.add_argument('--bal', help='BAL cut applied', default=None,required=False)
parser.add_argument('--suffix-flag', help='Optional suffix', default=None,required=False)

args = vars(parser.parse_args())

folder_name = bookkeeping.return_folder_name(args.region,
                                             args.snr_cut,
                                             args.lines_name,
                                             args.catalog_name,
                                             args.dla_name,
                                             args.bal,
                                             args.suffix_flag)
