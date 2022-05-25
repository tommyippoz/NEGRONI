from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.suod import SUOD

from negroni.connectors.PYODLearner import PYODLearner
from negroni.ensembles.StackingLearner import StackingLearner
from negroni.utils.negroni_utils import get_name


class SupUnsupStacker(StackingLearner):

    def __init__(self, outliers_fraction, meta_level_model, verbose=False):
        super().__init__(base_level_learners=[
                            PYODLearner(COPOD(contamination=outliers_fraction)),
                            PYODLearner(ABOD(contamination=outliers_fraction, method="fast")),
                            PYODLearner(HBOS(contamination=outliers_fraction)),
                            PYODLearner(MCD(contamination=outliers_fraction)),
                            PYODLearner(PCA(contamination=outliers_fraction, weighted=True)),
                            PYODLearner(ECOD(contamination=outliers_fraction)),
                            PYODLearner(LOF(contamination=outliers_fraction, n_jobs=-1)),
                            PYODLearner(CBLOF(contamination=outliers_fraction)),
                            PYODLearner(KNN(contamination=outliers_fraction)),
                            PYODLearner(IForest(contamination=outliers_fraction)),
                            PYODLearner(SUOD(contamination=outliers_fraction,
                                             base_estimators=[COPOD(), PCA(), CBLOF()]))],
                         meta_level_learner=meta_level_model,
                         use_training=False,
                         train_meta=False,
                         store_data=False,
                         verbose=verbose)

    def get_name(self):
        return "SupUnsupStacking(" + str(len(self.base_level_learners)) + "-" \
               + get_name(self.meta_level_learner) + ")"
