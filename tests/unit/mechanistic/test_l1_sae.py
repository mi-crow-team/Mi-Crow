
"""Tests for L1Sae module."""



import sys


import types


from unittest.mock import MagicMock


from pathlib import Path



import pytest


import torch




overcomplete_module = types.ModuleType('overcomplete')


overcomplete_sae_module = types.ModuleType('overcomplete.sae')



overcomplete_sae_module.SAE = MagicMock


overcomplete_module.SAE = MagicMock


overcomplete_module.TopKSAE = MagicMock


overcomplete_module.sae = overcomplete_sae_module


sys.modules['overcomplete'] = overcomplete_module


sys.modules['overcomplete.sae'] = overcomplete_sae_module



from mi_crow.hooks.hook import HookType


from mi_crow.mechanistic.sae.modules.l1_sae import L1Sae, L1SaeTrainingConfig




class StubEngine:


    """Stub engine that simulates OvercompleteSAE behavior."""


    def __init__(self):


        self.loaded = None



    def encode(self, x):




        pre_codes = x + 1


        codes = x * 0.5


        return pre_codes, codes



    def decode(self, z):


        return z + 2



    def forward(self, x):



        pre_codes, codes = self.encode(x)


        x_reconstructed = self.decode(codes)


        return pre_codes, codes, x_reconstructed



    def state_dict(self):


        return {"weight": torch.ones(1)}



    def load_state_dict(self, state):


        self.loaded = state



    def parameters(self):


        return [torch.nn.Parameter(torch.ones(1, requires_grad=True))]



    def to(self, *args, **kwargs):


        return self




@pytest.fixture(autouse=True)


def stub_engine(monkeypatch):


    """Replace OvercompleteSAE engine with stub for testing."""


    monkeypatch.setattr(
        L1Sae,
        "_initialize_sae_engine",
        lambda self: StubEngine(),
    )




def make_l1_sae():


    """Create an L1Sae instance for testing."""


    return L1Sae(n_latents=2, n_inputs=2)




def test_l1_sae_training_config():


    """Test L1SaeTrainingConfig initialization."""


    config = L1SaeTrainingConfig()


    assert config is not None



    assert hasattr(config, 'epochs')


    assert hasattr(config, 'batch_size')




def test_l1_sae_initialization():


    """Test L1Sae initialization."""


    sae = make_l1_sae()


    assert sae is not None


    assert sae.context.n_latents == 2


    assert sae.context.n_inputs == 2




def test_encode_decode_forward():


    """Test encode, decode, and forward methods."""


    sae = make_l1_sae()


    x = torch.ones(2, 2)



    encoded = sae.encode(x)


    assert torch.equal(encoded, x * 0.5)



    decoded = sae.decode(torch.zeros(2, 2))


    assert torch.equal(decoded, torch.full((2, 2), 2.0))



    forward_result = sae.forward(x)


    assert forward_result.shape == x.shape




def test_modify_activations_forward_hook():


    """Test modify_activations with FORWARD hook type."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD



    output = torch.ones(1, 2, 2)


    result = sae.modify_activations(module=None, inputs=(output,), output=output)



    assert result.shape == output.shape


    assert not torch.equal(result, output)




def test_modify_activations_pre_forward_hook():


    """Test modify_activations with PRE_FORWARD hook type."""


    sae = make_l1_sae()


    sae.hook_type = HookType.PRE_FORWARD



    input_tensor = torch.ones(1, 2)


    inputs = (input_tensor, torch.zeros(1, 2))


    result = sae.modify_activations(module=None, inputs=inputs, output=None)



    assert isinstance(result, tuple)


    assert result[0].shape == input_tensor.shape




def test_modify_activations_text_tracking():


    """Test modify_activations with text tracking enabled."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD


    sae._text_tracking_enabled = True



    tracker = MagicMock()


    tracker.get_current_texts.return_value = ["alpha"]



    class DummyLM:


        def get_input_tracker(self):


            return tracker



    sae.context.lm = DummyLM()


    spy = MagicMock()


    sae.concepts.update_top_texts_from_latents = spy



    output = torch.ones(1, 2, 2)


    sae.modify_activations(module=None, inputs=(output,), output=output)



    spy.assert_called_once()




def test_modify_activations_object_output():


    """Test modify_activations with object output (has last_hidden_state)."""


    class Wrapper:


        def __init__(self, tensor):


            self.last_hidden_state = tensor



    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD


    wrapper = Wrapper(torch.ones(1, 2))


    result = sae.modify_activations(module=None, inputs=(wrapper.last_hidden_state,), output=wrapper)



    assert isinstance(result, Wrapper)


    assert not torch.equal(result.last_hidden_state, torch.ones(1, 2))




def test_modify_activations_list_output():


    """Test modify_activations with list output."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD



    output = [torch.ones(1, 2), torch.zeros(1, 2)]


    result = sae.modify_activations(module=None, inputs=(), output=output)



    assert isinstance(result, list)


    assert len(result) == 2


    assert result[0].shape == (1, 2)




def test_modify_activations_tuple_output():


    """Test modify_activations with tuple output."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD



    output = (torch.ones(1, 2), torch.zeros(1, 2))


    result = sae.modify_activations(module=None, inputs=(), output=output)



    assert isinstance(result, tuple)


    assert len(result) == 2


    assert result[0].shape == (1, 2)




def test_modify_activations_3d_reshaping():


    """Test modify_activations with 3D input tensor."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD



    x = torch.randn(2, 3, 2)


    original_shape = x.shape



    result = sae.modify_activations(None, (), x)



    assert result.shape == original_shape


    assert result.shape == (2, 3, 2)




def test_modify_activations_concept_manipulation():


    """Test modify_activations with concept manipulation (multiplication/bias)."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD


    sae.concepts.multiplication.data = torch.full((2,), 2.0)


    sae.concepts.bias.data = torch.full((2,), 1.0)



    output = torch.ones(1, 2)


    result = sae.modify_activations(module=None, inputs=(), output=output)



    assert result.shape == output.shape




def test_modify_activations_saves_metadata():


    """Test that modify_activations saves metadata correctly."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD



    output = torch.ones(2, 2)


    sae.modify_activations(module=None, inputs=(), output=output)




    assert 'batch_items' in sae.metadata


    assert len(sae.metadata['batch_items']) == 2


    assert 'neurons' in sae.tensor_metadata


    assert 'activations' in sae.tensor_metadata




def test_modify_activations_no_tensor_returns_original():


    """Test modify_activations returns original when no tensor found."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD



    output = ["a", "b"]


    result = sae.modify_activations(module=None, inputs=(), output=output)


    assert result == output




def test_save_and_load(monkeypatch, tmp_path):


    """Test save and load methods."""


    captured = {}



    def fake_save(payload, path):


        captured["payload"] = payload



    monkeypatch.setattr("mi_crow.mechanistic.sae.modules.l1_sae.torch.save", fake_save)


    sae = make_l1_sae()


    sae.concepts.multiplication.data = torch.full((2,), 2.0)


    sae.save("model", path=tmp_path)



    assert "payload" in captured


    assert "sae_state_dict" in captured["payload"]


    assert "mi_crow_metadata" in captured["payload"]




    load_payload = {
        "sae_state_dict": {"weight": torch.ones(1)},
        "mi_crow_metadata": {
            "n_latents": 2,
            "n_inputs": 2,
            "device": "cpu",
            "concepts_state": {
                "multiplication": torch.full((2,), 3.0),
                "bias": torch.full((2,), 4.0),
            },
        },
    }


    monkeypatch.setattr("mi_crow.mechanistic.sae.modules.l1_sae.torch.load", lambda *args, **kwargs: load_payload)


    loaded = L1Sae.load(tmp_path / "model.pt")



    assert torch.equal(loaded.concepts.multiplication, torch.full((2,), 3.0))


    assert torch.equal(loaded.concepts.bias, torch.full((2,), 4.0))




def test_load_missing_metadata_raises(monkeypatch, tmp_path):


    """Test load raises ValueError when metadata is missing."""


    monkeypatch.setattr("mi_crow.mechanistic.sae.modules.l1_sae.torch.load", lambda *args, **kwargs: {})


    with pytest.raises(ValueError, match="missing 'mi_crow_metadata'"):


        L1Sae.load(tmp_path / "missing.pt")




def test_load_backward_compatibility_model_key(monkeypatch, tmp_path):


    """Test load with backward compatibility 'model' key."""


    load_payload = {
        "model": {"weight": torch.ones(1)},
        "mi_crow_metadata": {
            "n_latents": 2,
            "n_inputs": 2,
            "device": "cpu",
            "concepts_state": {},
        },
    }


    monkeypatch.setattr("mi_crow.mechanistic.sae.modules.l1_sae.torch.load", lambda *args, **kwargs: load_payload)


    loaded = L1Sae.load(tmp_path / "model.pt")


    assert loaded is not None




def test_load_backward_compatibility_state_dict(monkeypatch, tmp_path):


    """Test load with backward compatibility (payload is state dict)."""


    load_payload = {"weight": torch.ones(1)}


    monkeypatch.setattr("mi_crow.mechanistic.sae.modules.l1_sae.torch.load", lambda *args, **kwargs: load_payload)




    load_payload_with_meta = {
        "weight": torch.ones(1),
        "mi_crow_metadata": {
            "n_latents": 2,
            "n_inputs": 2,
            "device": "cpu",
            "concepts_state": {},
        },
    }


    monkeypatch.setattr("mi_crow.mechanistic.sae.modules.l1_sae.torch.load", lambda *args, **kwargs: load_payload_with_meta)


    loaded = L1Sae.load(tmp_path / "model.pt")


    assert loaded is not None




def test_save_and_load_roundtrip(tmp_path):


    """Test save and load roundtrip."""


    sae = make_l1_sae()


    sae.concepts.multiplication.data = torch.full((2,), 2.5)


    sae.concepts.bias.data = torch.full((2,), 1.5)


    sae.save("demo", tmp_path)



    loaded = L1Sae.load(tmp_path / "demo.pt")


    assert torch.allclose(loaded.concepts.multiplication, torch.full((2,), 2.5))


    assert torch.allclose(loaded.concepts.bias, torch.full((2,), 1.5))




def test_process_activations():


    """Test process_activations method (should do nothing)."""


    sae = make_l1_sae()



    sae.process_activations(module=None, input=(), output=torch.ones(1, 2))





def test_train_method(monkeypatch):


    """Test train method delegates to trainer."""


    sae = make_l1_sae()


    mock_trainer = MagicMock()


    mock_trainer.train.return_value = {"history": {"loss": [0.1]}, "training_run_id": "test_run"}


    sae.trainer = mock_trainer



    mock_store = MagicMock()


    result = sae.train(mock_store, "run_id", "layer_0", L1SaeTrainingConfig())



    mock_trainer.train.assert_called_once()


    assert "history" in result


    assert "training_run_id" in result




def test_train_method_default_config(monkeypatch):


    """Test train method with default config."""


    sae = make_l1_sae()


    mock_trainer = MagicMock()


    mock_trainer.train.return_value = {"history": {}, "training_run_id": "test"}


    sae.trainer = mock_trainer



    mock_store = MagicMock()


    result = sae.train(mock_store, "run_id", "layer_0")



    mock_trainer.train.assert_called_once()



    call_args = mock_trainer.train.call_args


    assert call_args[0][3] is not None




def test_modify_activations_pre_forward_3d():


    """Test PRE_FORWARD hook with 3D inputs."""


    sae = make_l1_sae()


    sae.hook_type = HookType.PRE_FORWARD



    x = torch.randn(2, 3, 2)


    original_shape = x.shape



    result = sae.modify_activations(None, (x,), None)



    assert isinstance(result, tuple)


    assert result[0].shape == original_shape


    assert result[0].shape == (2, 3, 2)




def test_modify_activations_saves_batch_items_metadata():


    """Test that modify_activations saves batch_items metadata correctly."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD



    output = torch.ones(3, 2)


    sae.modify_activations(module=None, inputs=(), output=output)



    assert 'batch_items' in sae.metadata


    assert len(sae.metadata['batch_items']) == 3


    for item in sae.metadata['batch_items']:


        assert 'nonzero_indices' in item


        assert 'activations' in item




def test_modify_activations_handles_none_tensor():


    """Test modify_activations handles None tensor gracefully."""


    sae = make_l1_sae()


    sae.hook_type = HookType.FORWARD




    output = None


    result = sae.modify_activations(module=None, inputs=(), output=output)


    assert result is None




def test_load_sets_context_metadata(monkeypatch, tmp_path):


    """Test that load sets context metadata correctly."""


    load_payload = {
        "sae_state_dict": {"weight": torch.ones(1)},
        "mi_crow_metadata": {
            "n_latents": 4,
            "n_inputs": 8,
            "device": "cpu",
            "layer_signature": "layer_5",
            "model_id": "test_model",
            "concepts_state": {},
        },
    }


    monkeypatch.setattr("mi_crow.mechanistic.sae.modules.l1_sae.torch.load", lambda *args, **kwargs: load_payload)


    loaded = L1Sae.load(tmp_path / "model.pt")



    assert loaded.context.lm_layer_signature == "layer_5"


    assert loaded.context.model_id == "test_model"


    assert loaded.context.n_latents == 4


    assert loaded.context.n_inputs == 8



