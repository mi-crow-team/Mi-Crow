
"""Tests for LayerActivationDetector."""



import pytest


import torch


from torch import nn



from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector


from mi_crow.hooks.hook import HookType




class TestLayerActivationDetectorInitialization:


    """Tests for LayerActivationDetector initialization."""



    def test_init_with_layer_signature_string(self):


        """Test initialization with string layer signature."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        assert detector.layer_signature == "layer_0"


        assert detector.hook_type == HookType.FORWARD



    def test_init_with_layer_signature_int(self):


        """Test initialization with int layer signature."""


        detector = LayerActivationDetector(layer_signature=0)


        assert detector.layer_signature == 0



    def test_init_with_custom_hook_id(self):


        """Test initialization with custom hook ID."""


        detector = LayerActivationDetector(layer_signature="layer_0", hook_id="custom_id")


        assert detector.id == "custom_id"



    def test_init_with_none_layer_signature_raises_error(self):


        """Test that None layer_signature raises ValueError."""


        with pytest.raises(ValueError, match="layer_signature cannot be None"):


            LayerActivationDetector(layer_signature=None)




class TestLayerActivationDetectorProcessActivations:


    """Tests for process_activations method."""



    def test_process_activations_single_tensor(self):


        """Test processing activations from single tensor output."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor = torch.randn(2, 5)



        detector.process_activations(module, (input_tensor,), output_tensor)



        captured = detector.get_captured()


        assert captured is not None


        assert torch.equal(captured, output_tensor.cpu())



    def test_process_activations_tuple_output(self):


        """Test processing activations from tuple output."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor1 = torch.randn(2, 5)


        output_tensor2 = torch.randn(2, 3)



        detector.process_activations(module, (input_tensor,), (output_tensor1, output_tensor2))



        captured = detector.get_captured()


        assert captured is not None


        assert torch.equal(captured, output_tensor1.cpu())



    def test_process_activations_with_last_hidden_state(self):


        """Test processing activations from object with last_hidden_state."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        hidden_tensor = torch.randn(2, 5)



        class Output:


            def __init__(self):


                self.last_hidden_state = hidden_tensor



        detector.process_activations(module, (input_tensor,), Output())



        captured = detector.get_captured()


        assert captured is not None


        assert torch.equal(captured, hidden_tensor.cpu())



    def test_process_activations_none_output(self):


        """Test processing activations when output is None."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)



        detector.process_activations(module, (input_tensor,), None)



        captured = detector.get_captured()


        assert captured is None



    def test_process_activations_overwrites_previous(self):


        """Test that processing new activations overwrites previous."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor1 = torch.randn(2, 5)


        output_tensor2 = torch.randn(3, 5)



        detector.process_activations(module, (input_tensor,), output_tensor1)


        captured1 = detector.get_captured()



        detector.process_activations(module, (input_tensor,), output_tensor2)


        captured2 = detector.get_captured()



        assert not torch.equal(captured1, captured2)


        assert torch.equal(captured2, output_tensor2.cpu())



    def test_process_activations_exception_handling(self):


        """Test that exceptions in process_activations are handled."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)





        detector.process_activations(module, (input_tensor,), "invalid")


        captured = detector.get_captured()


        assert captured is None



    def test_process_activations_cuda_tensor_moved_to_cpu(self):


        """Test that CUDA tensors are moved to CPU."""


        if not torch.cuda.is_available():


            pytest.skip("CUDA not available")



        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor = torch.randn(2, 5).cuda()



        detector.process_activations(module, (input_tensor,), output_tensor)



        captured = detector.get_captured()


        assert captured is not None


        assert captured.device.type == "cpu"


        assert torch.allclose(captured, output_tensor.cpu())



    def test_process_activations_preserves_gradient_detached(self):


        """Test that captured tensors are detached from computation graph."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10, requires_grad=True)


        output_tensor = torch.randn(2, 5, requires_grad=True)



        detector.process_activations(module, (input_tensor,), output_tensor)



        captured = detector.get_captured()


        assert captured is not None


        assert not captured.requires_grad



    def test_process_activations_list_output(self):


        """Test processing activations from list output."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor1 = torch.randn(2, 5)


        output_tensor2 = torch.randn(2, 3)



        detector.process_activations(module, (input_tensor,), [output_tensor1, output_tensor2])



        captured = detector.get_captured()


        assert captured is not None


        assert torch.equal(captured, output_tensor1.cpu())



    def test_process_activations_metadata_shape_stored(self):


        """Test that activation shape is stored in metadata."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor = torch.randn(3, 7, 5)



        detector.process_activations(module, (input_tensor,), output_tensor)



        assert 'activations_shape' in detector.metadata


        assert detector.metadata['activations_shape'] == (3, 7, 5)




class TestLayerActivationDetectorGetCaptured:


    """Tests for get_captured method."""



    def test_get_captured_returns_tensor(self):


        """Test that get_captured returns captured tensor."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor = torch.randn(2, 5)



        detector.process_activations(module, (input_tensor,), output_tensor)


        captured = detector.get_captured()



        assert captured is not None


        assert isinstance(captured, torch.Tensor)


        assert captured.device.type == "cpu"



    def test_get_captured_returns_none_when_nothing_captured(self):


        """Test that get_captured returns None when nothing captured."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        captured = detector.get_captured()


        assert captured is None




class TestLayerActivationDetectorClearCaptured:


    """Tests for clear_captured method."""



    def test_clear_captured_removes_tensor(self):


        """Test that clear_captured removes captured tensor."""


        detector = LayerActivationDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        input_tensor = torch.randn(2, 10)


        output_tensor = torch.randn(2, 5)



        detector.process_activations(module, (input_tensor,), output_tensor)


        assert detector.get_captured() is not None



        detector.clear_captured()


        assert detector.get_captured() is None



    def test_clear_captured_when_nothing_captured(self):


        """Test that clear_captured works when nothing captured."""


        detector = LayerActivationDetector(layer_signature="layer_0")



        detector.clear_captured()


        assert detector.get_captured() is None



