# Author: Nguyễn Thành Thủy, email: thuynt@due.edu.vn
# Trường Đại học Kinh tế, Đại học Đà Nẵng.
# Dự án Chatbot VIETNAM-AIRLINE-Assistant

from pyvi.pyvi import ViTokenizer, ViPosTagger
from sklearn.base import TransformerMixin, BaseEstimator


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = ViTokenizer()
        self.pos_tagger = ViPosTagger()

    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X.apply(lambda text: self.tokenizer.tokenize(text))
        return result
