import numpy as np

from models.blur_baseline import BlurBaseline
from models.center_bias import CenterBias


def test_model_contract():
    img = np.full((64, 96, 3), 128, dtype=np.uint8)
    for Model in (CenterBias, BlurBaseline):
        model = Model()
        x = model.preprocess(img)
        y = model.predict(x)
        y2 = model.postprocess(y, target_hw=(64, 96))
        assert y2.shape == (64, 96)
        assert np.isfinite(y2).all()
        assert (y2 >= 0).all()

