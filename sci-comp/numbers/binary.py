import struct

# This function has been adapted from Dan Lecocq's answer on Stack Overflow
# https://stackoverflow.com/a/16444778/2068994
def binary(num):
    # struct.pack provides us with the float packed into bytes. 
    # For single-precision, you could use 'f'.
    # ord() turns each charater into its corresponding integer code point
    # bin() converts it to its binary string representation.
    # replace('0b','') strips off the '0b' from each of these.
    # rjust() pads each byte's binary representation's with 0's to make sure it has all 8 bits.
    # join() concatenates them to get the total representation of the float
    return ','.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', num))

while True:
    num = float(input("Enter a number: "))
    print(binary(num))