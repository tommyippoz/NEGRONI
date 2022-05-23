from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class NEGRONILearner(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.classes_ = None
        self.X_ = None
        self.y_ = None

    def classifier_fit(self, train_features, train_labels):
        pass

    def classifier_predict_proba(self, test_features):
        pass

    def classifier_predict(self, test_features):
        pass

    def get_name(self):
        pass

    def fit(self, train_features, train_labels):
        # Check that X and y have correct shape
        train_features, y = check_X_y(train_features, train_labels)

        self.classifier_fit(train_features, train_labels)

        # Compatibility with SKLearn
        self.classes_ = unique_labels(y)
        self.X_ = train_features
        self.y_ = y

        return self

    def predict(self, test_features):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        test_features = check_array(test_features)

        return self.classifier_predict(test_features)

    def predict_proba(self, test_features):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        test_features = check_array(test_features)

        return self.classifier_predict_proba(test_features)
