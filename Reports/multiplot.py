import sys, os, json
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import curve_fit
from numpy import arctan, pi, log, exp

def poly(x, a, b):
    return a * np.power(x, b)

def quad_poly(x, a):
    return a * np.power(x, 2)

def plot_results(filenames):

	fig = figure(figsize=(9, 6))
	ax = fig.add_subplot(111)
	ax.set_yscale('log')
	ax.set_xscale('log')
	lightnavy = matplotlib.colors.colorConverter.to_rgb('#0174DF') 
	darknavy = matplotlib.colors.colorConverter.to_rgb('#084B8A')
	ax.yaxis.grid(True, linestyle='-', which='major', color='grey')
	ax.yaxis.grid(True, linestyle='-', which='minor', color='lightgrey')
	ax.xaxis.grid(True, linestyle='-', which='major', color='grey')
	ax.xaxis.grid(True, linestyle='-', which='minor', color='lightgrey')
	ax.set_xlabel('Physical fault rate')
	ax.set_ylabel('Logical fault rate')
	plt.rc('lines', linewidth= 2)
	plt.rc('lines', linestyle= None)
	plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))

	with open(filenames[0]) as data_file:
		res = json.load(data_file)
	lu_avg= [elt['res']['lu avg'] for elt in res]
	lu_std= [elt['res']['lu std'] for elt in res]
	for i in range(len(lu_std)):
		if lu_std[i]==0.00012393: lu_std[i]= 0.000012393
	upper= [elt['data']['fault scale'] for elt in res]
	p= [elt['res']['p'] for elt in res]
	p, lu_avg, lu_std, upper= zip(*sorted(zip(p, lu_avg, lu_std, upper)))
	g= ax.plot(p, lu_avg, \
		marker= 'o', markersize= 3, label= 'Look up table')
	g= ax.plot(p, quad_poly(p, *curve_fit(quad_poly, p, lu_avg)[0]), \
		color= g[-1].get_color(), linestyle= '--')
	ax.errorbar(p, lu_avg, yerr=lu_std, \
		color= g[-1].get_color(), linestyle='None', markersize=8, capsize=3)
	plt.plot(p, p, linestyle= '--', color= 'black', linewidth= 1)

	for file in filenames:
		with open(file) as data_file:
			res = json.load(data_file)
		nn_avg= [elt['res']['nn avg'] for elt in res]
		nn_std= [elt['res']['nn std'] for elt in res]
		p= [elt['res']['p'] for elt in res]
		p, nn_avg, nn_std = zip(*sorted(zip(p, nn_avg, nn_std)))
		label= res[0]['param']['nn']['type'] \
			 + str(len(res[0]['param']['nn']['num hidden']))
		g= ax.plot(p, nn_avg, \
			marker='o', markersize=3, label= label)
		nn_poly, _ = curve_fit(quad_poly, p, nn_avg)
		g= ax.plot(p, quad_poly(p, *nn_poly), \
			color= g[-1].get_color(), linestyle= '--')
		ax.errorbar(p, nn_avg, yerr=nn_std, \
			color= g[-1].get_color(), linestyle='None', markersize=8, capsize=3)

	ymin, ymax= ax.get_ylim()
	xmin, xmax= ax.get_xlim()
	plot_slope= log(ymax/ymin)/log(xmax/xmin)

	with open(filenames[0]) as data_file:
		res = json.load(data_file)
	lu_avg= [elt['res']['lu avg'] for elt in res]
	p= [elt['res']['p'] for elt in res]
	p, lu_avg = zip(*sorted(zip(p, lu_avg)))
	lu_poly, _ = curve_fit(quad_poly, p, lu_avg)
	ax.text(p[0], quad_poly(p[0], *lu_poly), \
		"$%f x^2$" % (lu_poly[0]), va= 'bottom', \
		fontsize=9, rotation= 180/ (pi * plot_slope) * arctan(2.0))

	for file in filenames:
		with open(file) as data_file:
			res = json.load(data_file)
		nn_avg= [elt['res']['nn avg'] for elt in res]
		p= [elt['res']['p'] for elt in res]
		p, nn_avg = zip(*sorted(zip(p, nn_avg)))
		nn_poly, _ = curve_fit(quad_poly, p, nn_avg)
		ax.text(p[0], quad_poly(p[0], *nn_poly), \
			"$%f x^2$" % (nn_poly[0]), va= 'bottom', \
			fontsize=9, rotation= 180/ (pi * plot_slope) * arctan(2.0))
 		
	plt.legend()
	plt.show()

if __name__ == '__main__':

	filenames= []
	for elt in os.listdir(sys.argv[1]):
		path= os.path.join(sys.argv[1], elt)
		if not os.path.isdir(path):
			filenames.append(path)
	print filenames
	plot_results(filenames)