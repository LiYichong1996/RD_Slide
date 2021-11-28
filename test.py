import RNA
import numpy as np

from utils.rna_lib import structure_dotB2Edge, structure_edge2DotB

dotB = '(((((((....(((...........)))((((((((..(((((((((((((((((((...(((((......))))).)))))).)))))))))))))..))))))))..)))))))'
edge_index = structure_dotB2Edge(dotB)
dotB_ = structure_edge2DotB(edge_index)

dist = RNA.hamming_distance(dotB_, dotB)

print(dist)