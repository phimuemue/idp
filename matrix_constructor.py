#/usr/bin/python

f=open("matrices.h","w")

import glob, os
import re

for mat in glob.glob("./*.matrix"):
    print ("Processing " +  mat + " ...")
    # collect stuff from file
    contents = []
    for line in open(mat):
        if line.lower().startswith("warning") or line.startswith(">"):
            continue
        parts = re.match(".*?(\[.*\]).*=(.*);",line).groups()
        index = [int(x) for x in re.search("\[(.*?)\]\[(.*)\]",parts[0]).groups()]
        value = parts[1]
        contents.append(index+[value,])
    # sort stuff
    maxx = max(contents, key=lambda x:x[0])[0]
    maxy = max(contents, key=lambda x:x[1])[1]
    print maxx
    print maxy
    realcontents = [[0 for y in xrange(maxy+1)] for x in xrange(maxx+1)]
    for [x,y,c] in contents:
        realcontents[x][y] = c
    # write stuff to matrices.h
    f.write ("T " + os.path.basename(mat).split(".matrix")[0] + "[%d][%d] = {\n"%(maxx+1,maxy+1))
    for i in realcontents:
        f.write("\t{" + ",".join(i) + "},\n")

    f.write("};\n")
f.close()



