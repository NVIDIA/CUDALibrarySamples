#!/usr/bin/env python3

# Copyright (c) 2018-2025 NVIDIA CORPORATION AND AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the NVIDIA CORPORATION nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import struct
import argparse
import numpy
import warnings

delimiter = ','

def fixInput(val):
    if val is None:
        return 0
    try:
        retval = int(val)
    except ValueError:
        retval = ord(val)

    return retval


if len(sys.argv) != 5 and len(sys.argv) != 6:
    print("Usage:")
    print("\tcsv_to_binary.py <input filename> <column choice> <datatype> <output filename> [delimiter]")
    print()
    print("This program converts one column of a text file containing a table of data,")
    print("(a comma-separated values file by default), into a binary file.")
    print()
    print("The <column choice> should be an integer in the range [0, N-1], where N is the number of columns.") 
    print("The <datatype> option should be one of 'int', 'long', 'float', 'double', or 'string'.")
    print("'string' keeps the text, converting it to UTF-16 with no separators between the values.")
    print("The [delimiter] is an optional argument, and defaults to '%s'" % delimiter)
    print("Some delimiters may need to be surrounded by quotation marks or prefixed by a backslash, depending on")
    print("the shell, for example space, semicolon, or vertical pipe, due to the command line parsing")
    print("interpreting the space or semicolon as a parameter separator or command separator, instead of a")
    print("parameter to this script.")
    print()
    print("Examples:")
    print("    text_to_binary.py ExampleFloatData.csv 2 float ZValues.bin")
    print("    text_to_binary.py ExampleTable.txt 5 long Dates.bin '|'")
    print("    text_to_binary.py SpaceSeparatedData.txt 0 int FirstColumn.bin ' '")
    print()
    exit()

in_fname = sys.argv[1]
col_num = sys.argv[2]
datatype = sys.argv[3]
out_fname = sys.argv[4]

if len(sys.argv) == 6:
    delimiter = sys.argv[5]

# Add more datatypes if needed
if datatype == "int":
    dtype = "int32"
elif datatype == "long":
    dtype = "int64"
elif datatype == "float":
    dtype = "float32"
elif datatype == "double":
    dtype = "float64"
elif datatype == "string":
    dtype = "str"
else:
    print("Please select datatype int, long, float, double, or string")
    exit()


print("Reading column " + col_num + ", of type " + datatype + "...")

chunk_size = 10000000
finished = False
offset = 0
with open(str(in_fname), "r") as inFile:
    with open(str(out_fname), "wb") as newFile:
        with warnings.catch_warnings():
            while not finished:
                in_data=numpy.genfromtxt(inFile, dtype=dtype,
                max_rows=chunk_size, usecols=(int(col_num),), delimiter=delimiter, loose=False)

                if offset == 0:
                    # don't warn about an empty file after we have read something
                    warnings.filterwarnings('ignore', r'genfromtxt: Empty input file:')

                if in_data.size > 0:
                    in_data.tofile(newFile)
                    offset += in_data.size
                else:
                    finished = True
if offset != 0:
    print('Wrote '+str(offset)+' '+datatype+'s to '+str(out_fname))
else:
    print('Wrote no data')
