import pickle


def pickle_save(obj, filename, outdir):
    with open(outdir + '/' + filename + '.pkl', 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
