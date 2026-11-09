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


def test_model_parameters():
    model = SimpleClassifier()

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    assert total_parameters > 0