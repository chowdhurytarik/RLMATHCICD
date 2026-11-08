import torch

from ml_project.model import SimpleClassifier


def test_model_output_shape():
    model = SimpleClassifier()

    x = torch.randn(4, 10)

    output = model(x)

    assert output.shape == (4, 2)


def test_model_output_finite():
    model = SimpleClassifier()

    x = torch.randn(4, 10)

    output = model(x)

    assert torch.isfinite(output).all()