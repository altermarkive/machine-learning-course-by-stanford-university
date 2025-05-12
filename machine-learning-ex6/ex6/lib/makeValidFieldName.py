#!/usr/bin/env python3


def makeValidFieldName(text):
    # From MATLAB doc: field names must begin with a letter, which may be
    # followed by any combination of letters, digits, and underscores.
    # Invalid characters will be converted to underscores, and the prefix
    # "x0x[Hex code]_" will be added if the first character is not a letter.
    # NOTE: The considerations listed above are irrelevant for Python.
    return text
