#!/usr/bin/env python3

import math

for n in range(1, 100):
    fifths = 3**n
    octaves = round(math.log2(fifths));
    error = math.fabs(fifths/2**octaves - 1)
    print(n, octaves, error)