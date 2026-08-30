"""A deliberately badly formatted file, kept as a demo target.

Ask the assistant: "how should samples/messy_style.py be formatted?"
It parses fine - the problems here are style and lint, not syntax.
"""
import sys
import os


def  average( numbers ):
    total=0
    for n in numbers :
        total = total+n
    return total/len( numbers )

class stats:
    def __init__(self,values):
        self.values=values
    def mean( self ):
        return average(self.values)
