
from ..predict.pipelines import OmicPipe

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from umap import UMAP


class OmicPCA(OmicPipe):

    def __init__(self):
        super().__init__([('norm', StandardScaler()), ('fit', PCA())])


class OmicTSNE(OmicPipe):

    def __init__(self):
        super().__init__([('norm', StandardScaler()), ('fit', TSNE())])


class OmicUMAP(OmicPipe):

    def __init__(self):
        super().__init__([('norm', StandardScaler()), ('fit', UMAP())])

