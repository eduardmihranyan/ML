from argparse import ArgumentParser
import sys

DESCRIPTION="Print header and first lines of input."
EXAMPLES="cat file.csv | csvhead -n 100"

def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)
    parser.add_argument('-n', '--number_of_lines', type=int, help='Number of first rows to print', default=10)
    parser.add_argument('-o', '--output_file', type=str, help='Output file. stdout is used by default')

    parser.add_argument('file', nargs='?', help='File to read input from. stdin is used by default')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    input_stream = open(args.file, 'r') if args.file else sys.stdin
    output_stream = open(args.output_file, 'w') if args.output_file else sys.stdout

    rows=''
    for i in range(args.number_of_lines+1):
        rows+=input_stream.readline()

    output_stream.write(rows)


    if input_stream != sys.stdin:
        input_stream.close()
    if output_stream != sys.stdout:
        output_stream.close()

if __name__ == '__main__':
    main()