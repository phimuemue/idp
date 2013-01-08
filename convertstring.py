import argparse, sys

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', default='convert_input.txt')
argparser.add_argument('-o', '--output', default='converted_output.txt')

args = argparser.parse_args()
infile = args.input
outfile = args.output

try:
    f = open(infile, 'r')
except IOError:
    sys.stderr.write("No input file %s found.\n" % infile)
    exit()
lines = []
for line in f:
    line = line.strip()
    if line == '' or line.startswith('>'):
        continue
    if line.startswith('"['):
        line = line[2:]
    if line.endswith(']"'):
        line = line[:-2]
    elif line.endswith('\\'):
        line = line[:-1]
    else:
        line = line+" "
    lines.append(line)
f.close()

exprlist = ''.join(lines).split(', ')

for i in range(len(exprlist)):
    exprlist[i] = exprlist[i].replace(' ', '')
    exprlist[i] = exprlist[i].replace('^', '**')

f = open(outfile, 'w')
f.write('\n'.join(exprlist))
f.close()
