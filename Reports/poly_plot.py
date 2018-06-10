import sys, os, json
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from scipy.optimize import curve_fit
from numpy import arctan, pi, log, exp

matplotlib.rcParams.update({'font.size': 14})

def latex_float(f):
    float_str = '{0:.2e}'.format(f)
    if 'e' in float_str:
        base, exponent = float_str.split('e')
        return '$' + r'{0}\! \times\! 10^{{{1}}}'.format(base, int(exponent)) + '$'
    else:
        return float_str

def write_poly(a, b):
	return ' (' + latex_float(a) + ' $p^{%.2f}$) ' % b

def poly(x, a, b):
    return a * np.power(x, b)

def plot_results(filenames, titles= None):

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
	plt.rc('lines', linewidth= 3)
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

	lu_poly, _ = curve_fit(poly, p, lu_avg)
	g= ax.plot(p, lu_avg, \
		marker='o', markersize=4, \
		label= 'Look up table' + write_poly(lu_poly[0], lu_poly[1]))
	ax.errorbar(p, lu_avg, yerr=lu_std, \
		color= g[-1].get_color(), linestyle='None', \
		markersize=9, capsize=4)
	g= ax.plot(p, poly(p, *lu_poly), \
		color= g[-1].get_color(), linestyle= '--')

	plt.plot(p, p, linestyle= '--', color= 'black', linewidth= 2)

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
		nn_poly, _ = curve_fit(poly, p, nn_avg)
		g= ax.plot(p, nn_avg, \
			marker='o', markersize=4, \
			label= label + write_poly(nn_poly[0], nn_poly[1]))
		ax.errorbar(p, nn_avg, yerr=nn_std, \
			color= g[-1].get_color(), linestyle='None', \
			markersize=9, capsize=4)
		g= ax.plot(p, poly(p, *nn_poly), \
			color= g[-1].get_color(), linestyle= '--')

	plt.legend()
	plt.show()

if __name__ == '__main__':

	poly_deg= int(sys.argv[1]) # dummy
	if os.path.isdir(sys.argv[1]):
		filenames= []
		for elt in os.listdir(sys.argv[2]):
			path= os.path.join(sys.argv[2], elt)
			if not os.path.isdir(path):
				filenames.append(path)
		print filenames
		plot_results(filenames)
	else:
		filenames= sys.argv[2::2]
		titles= sys.argv[3::2]
		plot_results(filenames, titles)
