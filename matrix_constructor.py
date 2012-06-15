#/usr/bin/python

f=open("matrices.h","w")

import glob, os

for mat in glob.glob("./*.matrix"):
    print ("Processing " +  mat + " ...")
    f.write (os.path.basename(mat).split("\.")[0] + " = {\n")
    for line in open(mat):
        f.write("\t{" + line.strip() + "},\n")
    f.write("};\n")
f.close()

