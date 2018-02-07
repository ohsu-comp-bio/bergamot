
from HetMan.features.pathways import *
import subprocess
import shlex

def get_regulators(intx_type):
    sif = load_sif()
    filt = sif.Type =='controls-expression-of'
    regulators = list(sif.loc[filt,:].UpGene.unique())
    return regulators

def main():
    sif = load_sif()
    filt = sif.Type =='controls-expression-of'
    regulators = list(sif.loc[filt,:].UpGene.unique())
    for r in regulators:
        subprocess.call(shlex.split('./run_test.sh {}'.format(r)

if __name__ == '__main__':
    print('submitting jobs')
    main()
