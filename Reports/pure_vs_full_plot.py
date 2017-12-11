import json
from pprint import pprint
import sys
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import curve_fit
from numpy import arctan, pi, log, exp

def poly(x, a, b):
    return a * np.power(x, b)

def quad_poly(x, a):
    return a * np.power(x, 2)

def plot_results(filename1, filename2):

	with open(filename1) as data_file:
	    res = json.load(data_file)
	nn_avg= [elt['res']['nn avg'] for elt in res]
	nn_std= [elt['res']['nn std'] for elt in res]
	lu_avg= [elt['res']['lu avg'] for elt in res]
	lu_std= [elt['res']['lu std'] for elt in res]
	upper= [elt['data']['fault scale'] for elt in res]
	p= [elt['res']['p'] for elt in res]
	p, lu_avg, lu_std, nn_avg, upper\
		= zip(*sorted(zip(p, lu_avg, lu_std, nn_avg, upper)))

	with open(filename2) as data_file:
	    res2 = json.load(data_file)
	pe_avg= [elt['res']['nn avg'] for elt in res2]
	pe_std= [elt['res']['nn std'] for elt in res2]
	q= [elt['res']['p'] for elt in res2]
	q, pe_avg= zip(*sorted(zip(q, pe_avg)))

	fig = figure(figsize=(9, 6))
	ax = fig.add_subplot(111)
	ax.set_yscale('log')
	ax.set_xscale('log')

	# Pick some colors
	lightgreen = matplotlib.colors.colorConverter.to_rgb('#7bb210') 
	darkgreen = matplotlib.colors.colorConverter.to_rgb('#669111')
	lightnavy = matplotlib.colors.colorConverter.to_rgb('#0174DF') 
	darknavy = matplotlib.colors.colorConverter.to_rgb('#084B8A')
	lightred = matplotlib.colors.colorConverter.to_rgb('#F7819F') 
	darkred = matplotlib.colors.colorConverter.to_rgb('#DF013A')
	verydarkred = matplotlib.colors.colorConverter.to_rgb('#B40431') 
 
	ax.errorbar(p, lu_avg, yerr=lu_std, \
		linestyle= 'None', color= darkred, markersize=8, capsize=3)
	ax.plot(p, lu_avg, \
		linestyle= 'None', marker= 'o', color= verydarkred, markersize= 3, \
		label= 'Look up table')
	ax.errorbar(p, nn_avg, yerr=nn_std, \
		linestyle= 'None', color= darknavy, markersize=8, capsize=3)
	plt.plot(p, nn_avg, \
		marker='o', linestyle = 'None', color= darknavy, markersize=3, \
		label= 'LU-based RNN')
	ax.errorbar(p, pe_avg, yerr=pe_std, \
		linestyle= 'None', color= darkgreen, markersize=8, capsize=3)
	plt.plot(p, pe_avg, \
		marker='o', linestyle = 'None', color= darkgreen, markersize=3, \
		label= 'PE-based RNN')

	ax.yaxis.grid(True, linestyle='-', which='major', color='grey')
	ax.yaxis.grid(True, linestyle='-', which='minor', color='lightgrey')
	ax.xaxis.grid(True, linestyle='-', which='major', color='grey')
	ax.xaxis.grid(True, linestyle='-', which='minor', color='lightgrey')

	lu_poly, _ = curve_fit(quad_poly, p, lu_avg)
	nn_poly, _ = curve_fit(quad_poly, p, nn_avg)
	pe_poly, _ = curve_fit(quad_poly, p, pe_avg)

	plt.plot(p, quad_poly(p, *lu_poly), linestyle= '--', color= lightred)
	plt.plot(p, quad_poly(p, *nn_poly), linestyle= '--', color= lightnavy)
	plt.plot(p, quad_poly(p, *pe_poly), linestyle= '--', color= lightgreen)
	plt.plot(p, p, linestyle= '--', color= 'black', linewidth= 1)
	# plt.plot(p, upper, linestyle= '--', color= 'grey')

	ymin, ymax= ax.get_ylim()
	xmin, xmax= ax.get_xlim()
	plot_slope= log(ymax/ymin)/log(xmax/xmin)

	ax.text(p[0], quad_poly(p[0], *lu_poly), \
		"$%f x^2$" % (lu_poly[0]), va= 'bottom', \
		fontsize=9, rotation= 180/ (pi * plot_slope) * arctan(2.0))
	ax.text(p[0], quad_poly(p[0], *nn_poly), \
		"$%f x^2$" % (nn_poly[0]), va= 'bottom', \
		fontsize=9, rotation= 180/ (pi * plot_slope) * arctan(2.0))
	ax.text(p[0], quad_poly(p[0], *pe_poly), \
		"$%f x^2$" % (pe_poly[0]), va= 'bottom', \
		fontsize=9, rotation= 180/ (pi * plot_slope) * arctan(2.0))

	ax.set_xlabel('Physical fault rate')
	ax.set_ylabel('Logical fault rate')
	ax.set_title(res[0]['param']['nn']['type'])
    # + ' ' + '-'.join([str(elt) for elt in res[0]['param']['nn']['activations']]
		
	plt.legend()
	plt.savefig(filename2+'_comparison.pdf')
	plt.show()

if __name__ == '__main__':

	plot_results(sys.argv[1], sys.argv[2])