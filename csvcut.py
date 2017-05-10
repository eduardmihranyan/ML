 #!/usr/bin/env python
from argparse import ArgumentParser
import sys


DESCRIPTION = 'csvcut - returns cvs file with only desired columns'
EXAMPLES = 'example: cat file.txt | csvcut -f "col_name"| less -SR'


def write_row(index ,row, output_stream, separator = ','):
    """
    Write_row modifeis the rows, leaving only the column desired
    :param index: the indeces of columns that need to remain in the file
    :param row: row represented as a list of columns
    :param output_stream: a stream of modified row

    """
    output_line = ''

    mi = 0 if index==[] else max(index) 

    for i, column in enumerate(row):
        if i not in index:
            continue
        output_line += column + separator if i < mi else  column + '\n' 
        
    output_stream.write(output_line)


def main():
    args = parse_args()
    input_stream = open(args.file, 'r') if args.file else sys.stdin
    output_stream = open(args.output_file, 'r') if args.output_file else sys.stdout

    columns = input_stream.readline().strip().split(args.separator)
    first_rows = [columns]
    
    index = []
    fields_int = []

    if len(args.fields) == 0:
        fields = []
    elif args.fields == []:
        fields = columns
    else:
        fields = args.fields[0].strip().split(args.separator)
    
        if ((fields[0]).isdigit()):
            for i in range (len(fields)):
                fields_int.append(int(fields[i]))



    c = list(range(len(columns)))
    unique_l = list(set(columns))
    unique_c = [columns.index(x) for x in unique_l]

    for i,row in enumerate(first_rows):
        if i == 0:
            if len(args.fields) == 0:
                index = []
            elif ((fields[0]).isdigit()):
                index = fields_int
            else:
                for j, column in enumerate(row):
                    if(column in fields):
                        index.append(j)

        if args.complement:
            index = [x for x in c if x not in index]
        
        if args.unique:
            index = [x for x in unique_c if x in index]

        write_row(index, row, output_stream, separator = args.separator)

    for row in input_stream:
        write_row(index, row.strip().split(args.separator), output_stream, separator = args.separator)
   
       
    if input_stream != sys.stdin:
        input_stream.close()
    if output_stream != sys.stdout:
        output_stream.close()

def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, epilog=EXAMPLES)

    parser.add_argument('-s', '--separator', type=str, help='Separator to be used', default=',')
    parser.add_argument('-o', '--output_file', type=str, help='Output file. stdout is used by default')
    parser.add_argument('-f', '--fields', nargs = '*', type=str, 
        help='fields to be left, please give the fields with the seperator, same as the dataset, and withou any spaces')
    parser.add_argument('-c', '--complement', help = 'leaves only the complement', action = 'store_true')
    parser.add_argument('-u', '--unique', help = 'Remove duplicates from list of FIELDS', action = 'store_true')    
    

    parser.add_argument('file', nargs='?', help='File to read input from. stdin is used by default')

    args = parser.parse_args()

    return args

main()