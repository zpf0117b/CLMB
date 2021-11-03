#!/usr/bin/python
import sys
import argparse

parser = argparse.ArgumentParser(
    description="""Command-line benchmark utility.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False)

parser.add_argument('vambpath', help='Path to vamb directory')
parser.add_argument('clusterspath', help='Path to clusters.tsv')
parser.add_argument('refpath', help='Path to reference file')
parser.add_argument('--tax', dest='taxpath', help='Path to taxonomic maps')
parser.add_argument('-m', dest='min_bin_size', metavar='', type=int,
                    default=200000, help='Minimum size of bins [200000]')
parser.add_argument('-s', dest='separator', help='Binsplit separator', default=None)
parser.add_argument('--disjoint', action='store_true', help='Enforce disjoint clusters')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

sys.path.append(args.vambpath)
import vamb
import os

# Check that files exist
for path in args.clusterspath, args.refpath, args.taxpath:
    if path is not None and not os.path.isfile(path):
        raise FileNotFoundError(path)

with open(args.clusterspath) as file:
    clusters = vamb.vambtools.read_clusters(file)

with open(args.refpath) as file:
    reference = vamb.benchmark.Reference.from_file(file)

if args.taxpath is not None:
    with open(args.taxpath) as file:
        reference.load_tax_file(file)

binning = vamb.benchmark.Binning(clusters, reference, minsize=args.min_bin_size, disjoint=args.disjoint,
                            binsplit_separator=args.separator)

for rank in range(len(binning.counters)):
    binning.print_matrix(rank)
    print("")

print('Vamb bins:')
for rank in binning.summary():
    print('\t'.join(map(str, rank)))


# import matplotlib.pyplot as plt

# for precision in 0.95, 0.9:
#     plt.figure(figsize=(10, 2))
#     colors = ['#DDDDDD', '#AAAAAA', '#777777', '#444444', '#000000']
#     recalls = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
#     for y, bins in zip((0, 1), (binning, binning)):
#         for color, recall in zip(colors, recalls):
#             plt.barh(y, bins.counters[1][(recall, precision)], color=color)

#     plt.title(str(precision), fontsize=18)
#     plt.yticks([0, 1], ['Vamb', 'MetaBAT2'], fontsize=16)
#     plt.xticks([i*25 for i in range(5)], fontsize=13)
#     plt.legend([str(i) for i in recalls], bbox_to_anchor=(1, 1.1), title='Recall', fontsize=12)
    
#     if precision == 0.9:
#         plt.xlabel('# of Genomes Identified', fontsize=16)
#     plt.gca().set_axisbelow(True)
#     plt.grid()

# plt.show()