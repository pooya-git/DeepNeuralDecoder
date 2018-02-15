import sys, os, json
import matplotlib.pyplot as plt
from pylab import *
from scipy.optimize import curve_fit
from numpy import arctan, pi, log, exp

def poly(x, a, b):
    return a * np.power(x, b)

def quad_poly(x, a):
    global poly_deg
    return a * np.power(x, poly_deg)

def plot_results(filenames, titles= None, shifts= None, global_shift= 0.0):

	global poly_deg
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
	plt.rc('lines', linestyle= 'none')
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

	if shifts:
		shift_iter= iter(shifts)
	if titles:
		title_iter= iter(titles)
	for file in filenames:
		with open(file) as data_file:
			res = json.load(data_file)
		nn_avg= [elt['res']['nn avg'] if 'nn avg' in elt['res'].keys() else \
			np.mean(elt['res']['nn res']) for elt in res]
		nn_std= [elt['res']['nn std'] if 'nn std' in elt['res'].keys() else \
			np.std(elt['res']['nn res']) for elt in res]
		p= [elt['res']['p'] for elt in res]
		p, nn_avg, nn_std = zip(*sorted(zip(p, nn_avg, nn_std)))
		if not titles:
			label= res[0]['param']['nn']['type'] \
				 + str(len(res[0]['param']['nn']['num hidden']))
		else:
			label= next(title_iter)
		g= ax.plot(p, nn_avg, \
			marker='o', markersize=3, label= label)
		ax.errorbar(p, nn_avg, yerr=nn_std, \
			color= g[-1].get_color(), linestyle='None', \
			markersize=8, capsize=3)
		if (not shifts) or (shifts and not next(shift_iter)=='no'):
			nn_poly, _ = curve_fit(quad_poly, p, nn_avg)
			g= ax.plot(p, quad_poly(p, *nn_poly), \
				color= g[-1].get_color(), linestyle= '--')

	ymin, ymax= ax.get_ylim()
	xmin, xmax= ax.get_xlim()
	plot_slope= log(ymax/ymin)/log(xmax/xmin)

	with open(filenames[0]) as data_file:
		res = json.load(data_file)
	lu_avg= [elt['res']['lu avg'] for elt in res]
	p= [elt['res']['p'] for elt in res]
	p, lu_avg = zip(*sorted(zip(p, lu_avg)))
	lu_poly, _ = curve_fit(quad_poly, p, lu_avg)
	ax.text(p[0]+0.000005, quad_poly(p[0]+0.000005, *lu_poly) + global_shift, \
		"$%.2e x^%d$" % (lu_poly[0], poly_deg), va= 'bottom', \
		fontsize=9, rotation= 260 / (pi * plot_slope) * arctan(3.0))

	if shifts:
		shift_iter= iter(shifts)
	for file in filenames:
		with open(file) as data_file:
			res = json.load(data_file)
		nn_avg= [elt['res']['nn avg'] if 'nn avg' in elt['res'].keys() else \
			np.mean(elt['res']['nn res']) for elt in res]
		p= [elt['res']['p'] for elt in res]
		p, nn_avg = zip(*sorted(zip(p, nn_avg)))
		nn_poly, _ = curve_fit(quad_poly, p, nn_avg)
		if shifts:
			this_shift= next(shift_iter)
		ax.text(p[0]+0.000005, quad_poly(p[0]+0.000005, *nn_poly) \
				+ global_shift + float(this_shift) \
								 if shifts and not this_shift=='no' \
								 else 0.0, \
			"$%.2e x^%d$" % (nn_poly[0], poly_deg), va= 'bottom', \
			fontsize=9, rotation= 260 / (pi * plot_slope) * arctan(3.0))
 		
	plt.legend()
	plt.show()

if __name__ == '__main__':

	global_shift= None
	poly_deg= 2
	if (sys.argv[1]):
		global_shift= float(sys.argv[1])
	if (sys.argv[2]):
		poly_deg= int(sys.argv[2])
	global poly_deg
	if os.path.isdir(sys.argv[3]):
		filenames= []
		for elt in os.listdir(sys.argv[3]):
			path= os.path.join(sys.argv[3], elt)
			if not os.path.isdir(path):
				filenames.append(path)
		print filenames
		plot_results(filenames, global_shift= global_shift)
	else:
		filenames= sys.argv[3::3]
		titles= sys.argv[4::3]
		shifts= sys.argv[5::3]		
		plot_results(filenames, titles, shifts, global_shift)
