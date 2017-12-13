
import sys
import os


# imports ophion client for interacting with BMEG programatically
sys.path.extend(['../../ophion/client/python/'])
sys.path.extend(['/home/exacloud/lustre1/CompBio/estabroj/bergamot/ophion/client/python/'])

# gets directory where features that cannot be downloaded programatically
# are stored
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')

__all__ = ['cohorts']
