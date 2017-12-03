import sys, os, json
from time import time, strftime, localtime
import cPickle as pickle
from ExRecCNOT import *
from Surface1EC import *

def run_pickler(spec, param):

    for filename in os.listdir(param['env']['raw folder']):

        with open(param['env']['pickle folder'] + \
            filename.replace('.txt', '.pkl'), "wb") as output_file:
            print("Reading data from " + filename)
            if (param['env']['FT scheme']=='ExRecCNOT'):
                model= ExRecCNOT(param['env']['raw folder']+ filename, spec)
            elif (param['env']['FT scheme']=='Surface1EC'):
                model= Surface1EC(param['env']['raw folder']+ filename, spec)
            else:
                raise ValueError('Unknown circuit type.')
            pickle.dump(model, output_file)

def run_benchmark(spec, param):

    output= []
    for filename in os.listdir(param['env']['pickle folder']):

        with open(param['env']['pickle folder'] + filename, 'rb') as input_file:
            start_time= time()
            print('Pickling model from ' + filename + ' ...'),
            m = pickle.load(input_file)
            print('Done in ' + '{0:.2f}'.format(time() - start_time) + 's.')

        m.test_size= int(param['data']['test fraction'] * m.data_size)
        m.train_size= m.data_size - m.test_size
        m.num_batches= m.train_size // param['opt']['batch size']
        m.spec= spec

        fault_rates= []
        for i in range(param['data']['num trials']):
            prediction, test_beg= m.train(param, i)
            print('Testing ...'),
            start_time= time()
            result= m.num_logical_fault(prediction, test_beg)
            print('Done in ' + '{0:.2f}'.format(time() - start_time) + 's.')
            print m.error_scale * result
            fault_rates.append(m.error_scale * result)

        run_log= {}
        run_log['data']= {}
        run_log['opt']= {}
        run_log['res']= {}
        run_log['param']= param
        run_log['data']['path']= filename
        run_log['data']['fault scale']= m.error_scale
        run_log['data']['total size']= m.total_size
        run_log['data']['test size']= m.test_size
        run_log['data']['train size']= m.train_size
        run_log['opt']['batch size']= param['opt']['batch size']
        run_log['opt']['number of batches']= m.num_batches
        run_log['res']['p']= m.p
        run_log['res']['lu avg']= m.lu_avg
        run_log['res']['lu std']= m.lu_std
        run_log['res']['nn res'] = fault_rates
        run_log['res']['nn avg'] = np.mean(fault_rates)
        run_log['res']['nn std'] = np.std(fault_rates)
        output.append(run_log)

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    f = open(param['env']['report folder'] + outfilename + '.json', 'w')
    f.write(json.dumps(output, indent=2))
    f.close()

if __name__=='__main__':

    with open(sys.argv[2]) as paramfile:
        param = json.load(paramfile)

    if(param['env']['EC scheme']=='SurfaceD3'):
        import _SurfaceD3Lookup as lookup 
    elif(param['env']['EC scheme']=='SurfaceD5'):
        import _SurfaceD5Lookup as lookup 
    elif(param['env']['EC scheme']=='ColorD3'):
        import _ColorD3Lookup as lookup 
    elif(param['env']['EC scheme']=='ColorD5'):
        import _ColorD5Lookup as lookup 
    else:
        raise ValueError('Unknown circuit type.')
    spec= lookup.Spec()

    if (sys.argv[1]=='gen'):
        run_pickler(spec, param)

    elif (sys.argv[1]=='bench'):
        run_benchmark(spec, param)
    else:
        print('Error: Unrecognized flag!')
