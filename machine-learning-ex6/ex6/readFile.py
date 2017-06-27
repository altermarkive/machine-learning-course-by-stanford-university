#!/usr/bin/env python3


def readFile(filename):
    #READFILE reads a file and returns its entire contents 
    #   file_contents = READFILE(filename) reads a file and returns its entire
    #   contents in file_contents
    #

    # Load File
    with open(filename) as fid:
        file_contents = fid.read()
        #end
    return file_contents
    #end

