#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter GWAS result CSVs and append significant hits to an output file."
    )
    parser.add_argument(
        "glob_pattern",
        help="Quoted glob pattern for input CSV files (e.g. 'output/gwas/**/*_all_results.csv')",
    )
    parser.add_argument("output_file", help="Path to the output CSV file to append results to")
    args = parser.parse_args()

    input_files = sorted(glob.glob(args.glob_pattern, recursive=True))
    if not input_files:
        print(f"No files found matching: {args.glob_pattern}", file=sys.stderr)
        sys.exit(1)

    write_header = not os.path.exists(args.output_file)

    for filepath in input_files:
        basename = os.path.basename(filepath)
        sep = '_'
        embedding = basename.split(sep)[1:4]
        embedding = sep.join(embedding)
        df = pd.read_csv(filepath)
        df["embedding"] = embedding
        df.to_csv(args.output_file, mode="a", header = write_header, index=False)
        write_header = False


    print(f"Done. Results written to {args.output_file}")


if __name__ == "__main__":
    main()
