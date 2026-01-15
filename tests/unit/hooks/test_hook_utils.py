
"""Tests for hook utility functions."""



import pytest


import torch


from torch import nn



from mi_crow.hooks.utils import extract_tensor_from_input, extract_tensor_from_output




class TestExtractTensorFromInput:


    """Tests for extract_tensor_from_input function."""



    def test_extract_tensor_from_input_single_tensor(self):


        """Test extracting tensor from single tensor input."""


        tensor = torch.randn(2, 10)


        result = extract_tensor_from_input((tensor,))


        assert result is not None


        assert torch.equal(result, tensor)



    def test_extract_tensor_from_input_multiple_tensors(self):


        """Test extracting tensor from multiple tensor input."""


        tensor1 = torch.randn(2, 10)


        tensor2 = torch.randn(2, 5)


        result = extract_tensor_from_input((tensor1, tensor2))


        assert result is not None


        assert torch.equal(result, tensor1)



    def test_extract_tensor_from_input_empty_tuple(self):


        """Test extracting tensor from empty input."""


        result = extract_tensor_from_input(())


        assert result is None



    def test_extract_tensor_from_input_none(self):


        """Test extracting tensor from None input."""


        result = extract_tensor_from_input(None)


        assert result is None



    def test_extract_tensor_from_input_list(self):


        """Test extracting tensor from list input."""


        tensor = torch.randn(2, 10)


        result = extract_tensor_from_input([tensor])


        assert result is not None


        assert torch.equal(result, tensor)



    def test_extract_tensor_from_input_nested_tuple(self):


        """Test extracting tensor from nested tuple input."""


        tensor = torch.randn(2, 10)


        result = extract_tensor_from_input(((tensor,),))


        assert result is not None


        assert torch.equal(result, tensor)



    def test_extract_tensor_from_input_no_tensor(self):


        """Test extracting tensor when no tensor in input."""


        result = extract_tensor_from_input(("string", 123))


        assert result is None




class TestExtractTensorFromOutput:


    """Tests for extract_tensor_from_output function."""



    def test_extract_tensor_from_output_single_tensor(self):


        """Test extracting tensor from single tensor output."""


        tensor = torch.randn(2, 10)


        result = extract_tensor_from_output(tensor)


        assert result is not None


        assert torch.equal(result, tensor)



    def test_extract_tensor_from_output_tuple(self):


        """Test extracting tensor from tuple output."""


        tensor1 = torch.randn(2, 10)


        tensor2 = torch.randn(2, 5)


        result = extract_tensor_from_output((tensor1, tensor2))


        assert result is not None


        assert torch.equal(result, tensor1)



    def test_extract_tensor_from_output_list(self):


        """Test extracting tensor from list output."""


        tensor = torch.randn(2, 10)


        result = extract_tensor_from_output([tensor])


        assert result is not None


        assert torch.equal(result, tensor)



    def test_extract_tensor_from_output_none(self):


        """Test extracting tensor from None output."""


        result = extract_tensor_from_output(None)


        assert result is None



    def test_extract_tensor_from_output_with_last_hidden_state(self):


        """Test extracting tensor from object with last_hidden_state."""


        tensor = torch.randn(2, 10)



        class Output:


            def __init__(self):


                self.last_hidden_state = tensor



        result = extract_tensor_from_output(Output())


        assert result is not None


        assert torch.equal(result, tensor)



    def test_extract_tensor_from_output_with_invalid_last_hidden_state(self):


        """Test extracting tensor when last_hidden_state is not a tensor."""


        class Output:


            def __init__(self):


                self.last_hidden_state = "not a tensor"



        result = extract_tensor_from_output(Output())


        assert result is None



    def test_extract_tensor_from_output_no_tensor(self):


        """Test extracting tensor when no tensor in output."""


        result = extract_tensor_from_output("string")


        assert result is None



    def test_extract_tensor_from_output_mixed_types(self):


        """Test extracting tensor from mixed type output."""


        tensor = torch.randn(2, 10)


        result = extract_tensor_from_output((tensor, "string", 123))


        assert result is not None


        assert torch.equal(result, tensor)



