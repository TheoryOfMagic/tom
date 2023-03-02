# tom
Core "Theory of Magic" package.

# SpellWritingGuide
The messy *messy* Code that makes the spell writing guide

Copying from the message I sent to discord user Shparki#5522

The rough outline is:

```1) Make N points, (by default points on a regular polygon at angles 2pi/N), storing their locations in cartesian space

2) Create rotationally symmetric binary numbers of N-bits (I just load these in, I have the files if you need them up to N = 13 due to long computation time), 

3) Assign each feature of every attribute one of these binary numbers. (i.e. if we have a fire spell in [ice, water, fire] we assign it the 3rd binary number, if we had an ice spell it gets the 1st) We reserve the 0th binary ([0,0,....,0]) for an empty (i.e. no damage type)
```
> Important note, attribute is something like damage type, level, range etc. Feature is the specific element in the set that we use. I.e. Ice is a feature of the damage type attribute
```4) Assign each attribute a value "k (>=1)" which can also be called order. You can have (N/2)-1 attributes

5) To decode the binary numbers to lines its essentially just "point i-> i+k has a line if binary_number[i] == 1, no line (or feint dotted line) if binary_number[i] == 0
N.B. the final point connects to the first + k

6) Do that for every point```
