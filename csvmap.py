from argparse import ArgumentParser
import sys
import re
from collections import defaultdict


DESCRIPTION = 'csvmap - transform each row of a csv file with an expression provided.'
EXAMPLES = "example: cat file.csv | csvmap 'r.ratio = r.a / r.b, r.b *= 1000'"

def write_row(row, output_stream, separator):

    output_line = ''

    for i, column in enumerate(row):
        output_line += str(column) + separator 
    output_line += '\n'
    output_stream.write(output_line)

def main():
    args = parse_args()
    input_stream = open(args.file, 'r') if args.file else sys.stdin
    output_stream = open(args.output_file, 'r') if args.output_file else sys.stdout

    if args.exec != 'NONE':
        exec(args.exec)


    expressions = (args.expression.split(args.separator))
    columns = input_stream.readline().strip().split(args.separator)

    
    dicts = []

    for row in input_stream:
        row =[int(x) if x.isdigit() else x for x in row.strip().split(args.separator)]
        dictionary = dict(zip(columns,row))

        dicts.append(dictionary)

    for dictionary in dicts:
        C = type('type_C', (object,), dictionary)
        r = C()
        for expression in expressions:
            exec(expression)
        dictionary.update(r.__dict__)

    first_rows = list(dicts[0].keys())
    write_row(first_rows, output_stream, separator = args.separator)
    for ind in range(len(dicts)):
        current = (dicts[ind].values())
        write_row(current, output_stream, separator = args.separator)
               
       
    if input_stream != sys.stdin:
        input_stream.close()
    if output_stream != sys.stdout:
        output_stream.close()

def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)

    parser.add_argument('-s', '--separator', type=str, help='Separator to be used', default=',')
    parser.add_argument('-o', '--output_file', type=str, help='Output file. stdout is used by default')
    parser.add_argument('-e', '--exec', type = str, help = 'Execute python code before starting the transformation', default='NONE')  
    
    parser.add_argument('expression', help = 'Python expression to be used to transform a row')

    parser.add_argument('file', nargs='?', help='File to read input from. stdin is used by default')
    args = parser.parse_args()

    return args

main()