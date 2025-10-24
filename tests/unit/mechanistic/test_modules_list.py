from amber.mechanistic.autoencoder.modules.modules_list import get_activation
from amber.mechanistic.autoencoder.modules.topk import TopK


def test_get_activation_variants():
    act = get_activation("TopK_3")
    assert isinstance(act, TopK)
    act2 = get_activation("TopKReLU_2")
    assert isinstance(act2, TopK)
    # exercise extra_repr
    s = act2.extra_repr()
    assert "k=" in s
