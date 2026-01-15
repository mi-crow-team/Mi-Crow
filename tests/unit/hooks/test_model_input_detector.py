
"""Tests for ModelInputDetector."""



import pytest


import torch


from torch import nn


from unittest.mock import MagicMock, patch



from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector


from mi_crow.hooks.hook import HookType




class TestModelInputDetectorInitialization:


    """Tests for ModelInputDetector initialization."""



    @pytest.mark.parametrize("layer_signature", ["layer_0", 0, None, "model_inputs"])


    def test_init_with_various_layer_signatures(self, layer_signature):


        """Test initialization with various layer signature types."""


        detector = ModelInputDetector(layer_signature=layer_signature)


        assert detector.layer_signature == layer_signature


        assert detector.hook_type == HookType.PRE_FORWARD


        assert detector.save_input_ids is True


        assert detector.save_attention_mask is False



    def test_init_with_custom_hook_id(self):


        """Test initialization with custom hook ID."""


        detector = ModelInputDetector(layer_signature="layer_0", hook_id="custom_id")


        assert detector.id == "custom_id"



    @pytest.mark.parametrize("save_input_ids,save_attention_mask", [
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ])


    def test_init_with_save_flags(self, save_input_ids, save_attention_mask):


        """Test initialization with different save flag combinations."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_input_ids=save_input_ids,
            save_attention_mask=save_attention_mask
        )


        assert detector.save_input_ids == save_input_ids


        assert detector.save_attention_mask == save_attention_mask




class TestModelInputDetectorExtractInputIds:


    """Tests for _extract_input_ids method."""



    def test_extract_input_ids_from_dict(self):


        """Test extracting input_ids from dict input."""


        detector = ModelInputDetector(layer_signature="layer_0")


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        input_data = ({"input_ids": input_ids},)



        result = detector._extract_input_ids(input_data)



        assert result is not None


        assert torch.equal(result, input_ids)



    def test_extract_input_ids_from_dict_missing_key(self):


        """Test extracting input_ids when dict doesn't have input_ids key."""


        detector = ModelInputDetector(layer_signature="layer_0")


        input_data = ({"attention_mask": torch.ones(2, 3)},)



        result = detector._extract_input_ids(input_data)



        assert result is None



    def test_extract_input_ids_from_tensor(self):


        """Test extracting input_ids from direct tensor input."""


        detector = ModelInputDetector(layer_signature="layer_0")


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        input_data = (input_ids,)



        result = detector._extract_input_ids(input_data)



        assert result is not None


        assert torch.equal(result, input_ids)



    def test_extract_input_ids_empty_input(self):


        """Test extracting input_ids from empty input."""


        detector = ModelInputDetector(layer_signature="layer_0")



        result = detector._extract_input_ids(())


        assert result is None



        result = detector._extract_input_ids([])


        assert result is None



    def test_extract_input_ids_unsupported_type(self):


        """Test extracting input_ids from unsupported input type."""


        detector = ModelInputDetector(layer_signature="layer_0")


        input_data = ("invalid",)



        result = detector._extract_input_ids(input_data)


        assert result is None




class TestModelInputDetectorExtractAttentionMask:


    """Tests for _extract_attention_mask method."""



    def test_extract_attention_mask_from_dict(self):


        """Test extracting attention_mask from dict input."""


        detector = ModelInputDetector(layer_signature="layer_0")


        attention_mask = torch.ones(2, 3)


        input_data = ({"attention_mask": attention_mask},)



        result = detector._extract_attention_mask(input_data)



        assert result is not None


        assert torch.equal(result, attention_mask)



    def test_extract_attention_mask_missing_key(self):


        """Test extracting attention_mask when dict doesn't have the key."""


        detector = ModelInputDetector(layer_signature="layer_0")


        input_data = ({"input_ids": torch.ones(2, 3)},)



        result = detector._extract_attention_mask(input_data)


        assert result is None



    def test_extract_attention_mask_empty_input(self):


        """Test extracting attention_mask from empty input."""


        detector = ModelInputDetector(layer_signature="layer_0")



        result = detector._extract_attention_mask(())


        assert result is None



    def test_extract_attention_mask_non_dict_first_item(self):


        """Test extracting attention_mask when first item is not a dict."""


        detector = ModelInputDetector(layer_signature="layer_0")


        input_data = (torch.tensor([1, 2, 3]),)



        result = detector._extract_attention_mask(input_data)


        assert result is None




class TestModelInputDetectorSetInputsFromEncodings:


    """Tests for set_inputs_from_encodings method."""



    def test_set_inputs_from_encodings_with_input_ids(self):


        """Test setting inputs from encodings with input_ids."""


        detector = ModelInputDetector(layer_signature="layer_0", save_input_ids=True)


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        encodings = {"input_ids": input_ids}



        detector.set_inputs_from_encodings(encodings)



        captured = detector.get_captured_input_ids()


        assert captured is not None


        assert torch.equal(captured, input_ids.cpu())


        assert captured.device.type == "cpu"


        assert detector.metadata['input_ids_shape'] == tuple(input_ids.shape)



    def test_set_inputs_from_encodings_with_attention_mask(self):


        """Test setting inputs from encodings with attention_mask."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_attention_mask=True
        )


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        attention_mask = torch.ones(2, 3)


        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}



        detector.set_inputs_from_encodings(encodings)



        captured = detector.get_captured_attention_mask()


        assert captured is not None


        assert captured.shape == input_ids.shape


        assert detector.metadata['attention_mask_shape'] == tuple(captured.shape)



    def test_set_inputs_from_encodings_with_both(self):


        """Test setting inputs from encodings with both input_ids and attention_mask."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_input_ids=True,
            save_attention_mask=True
        )


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        attention_mask = torch.ones(2, 3)


        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}



        detector.set_inputs_from_encodings(encodings)



        assert detector.get_captured_input_ids() is not None


        assert detector.get_captured_attention_mask() is not None



    def test_set_inputs_from_encodings_save_input_ids_false(self):


        """Test that input_ids are not saved when save_input_ids is False."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_input_ids=False
        )


        encodings = {"input_ids": torch.tensor([[1, 2, 3]])}



        detector.set_inputs_from_encodings(encodings)



        assert detector.get_captured_input_ids() is None



    def test_set_inputs_from_encodings_save_attention_mask_false(self):


        """Test that attention_mask is not saved when save_attention_mask is False."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_attention_mask=False
        )


        encodings = {"attention_mask": torch.ones(2, 3)}



        detector.set_inputs_from_encodings(encodings)



        assert detector.get_captured_attention_mask() is None



    def test_set_inputs_from_encodings_missing_keys(self):


        """Test setting inputs when encodings don't have required keys."""


        detector = ModelInputDetector(layer_signature="layer_0")


        encodings = {}




        detector.set_inputs_from_encodings(encodings)



        assert detector.get_captured_input_ids() is None


        assert detector.get_captured_attention_mask() is None



    def test_set_inputs_from_encodings_exception_handling(self):


        """Test exception handling in set_inputs_from_encodings."""


        detector = ModelInputDetector(layer_signature="layer_0")




        with patch.object(torch.Tensor, 'detach', side_effect=RuntimeError("Test error")):


            encodings = {"input_ids": torch.tensor([[1, 2, 3]])}



            with pytest.raises(RuntimeError, match="Error setting inputs from encodings"):


                detector.set_inputs_from_encodings(encodings)




class TestModelInputDetectorProcessActivations:


    """Tests for process_activations method."""



    def test_process_activations_with_dict_input(self):


        """Test processing activations with dict input containing input_ids."""


        detector = ModelInputDetector(layer_signature="layer_0", save_input_ids=True)


        module = nn.Linear(10, 5)


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        input_data = ({"input_ids": input_ids},)



        detector.process_activations(module, input_data, None)



        captured = detector.get_captured_input_ids()


        assert captured is not None


        assert torch.equal(captured, input_ids.cpu())


        assert detector.metadata['input_ids_shape'] == tuple(input_ids.shape)



    def test_process_activations_with_tensor_input(self):


        """Test processing activations with direct tensor input."""


        detector = ModelInputDetector(layer_signature="layer_0", save_input_ids=True)


        module = nn.Linear(10, 5)


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        input_data = (input_ids,)



        detector.process_activations(module, input_data, None)



        captured = detector.get_captured_input_ids()


        assert captured is not None


        assert torch.equal(captured, input_ids.cpu())



    def test_process_activations_with_attention_mask(self):


        """Test processing activations with attention_mask."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_attention_mask=True
        )


        module = nn.Linear(10, 5)


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        attention_mask = torch.ones(2, 3)


        input_data = ({"input_ids": input_ids, "attention_mask": attention_mask},)



        detector.process_activations(module, input_data, None)



        captured = detector.get_captured_attention_mask()


        assert captured is not None


        assert captured.shape == input_ids.shape



    def test_process_activations_empty_input(self):


        """Test processing activations with empty input."""


        detector = ModelInputDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)



        detector.process_activations(module, (), None)



        assert detector.get_captured_input_ids() is None


        assert detector.get_captured_attention_mask() is None



    def test_process_activations_overwrites_previous(self):


        """Test that processing new activations overwrites previous."""


        detector = ModelInputDetector(layer_signature="layer_0", save_input_ids=True)


        module = nn.Linear(10, 5)


        input_ids1 = torch.tensor([[1, 2, 3]])


        input_ids2 = torch.tensor([[4, 5, 6], [7, 8, 9]])



        detector.process_activations(module, ({"input_ids": input_ids1},), None)


        captured1 = detector.get_captured_input_ids()



        detector.process_activations(module, ({"input_ids": input_ids2},), None)


        captured2 = detector.get_captured_input_ids()



        assert not torch.equal(captured1, captured2)


        assert torch.equal(captured2, input_ids2.cpu())



    def test_process_activations_exception_handling(self):


        """Test exception handling in process_activations."""


        detector = ModelInputDetector(layer_signature="layer_0", save_input_ids=True)


        module = nn.Linear(10, 5)




        with patch.object(detector, '_extract_input_ids', side_effect=RuntimeError("Test error")):


            with pytest.raises(RuntimeError, match="Error extracting inputs"):


                detector.process_activations(module, ({"input_ids": torch.tensor([1, 2, 3])},), None)



    def test_process_activations_input_ids_none_does_not_save(self):


        """Test that None input_ids are not saved."""


        detector = ModelInputDetector(layer_signature="layer_0", save_input_ids=True)


        module = nn.Linear(10, 5)



        input_data = ({"other_key": torch.tensor([1, 2, 3])},)



        detector.process_activations(module, input_data, None)



        assert detector.get_captured_input_ids() is None



    def test_process_activations_attention_mask_none_creates_default_mask(self):


        """Test that when attention_mask is None, a default mask is created."""


        detector = ModelInputDetector(layer_signature="layer_0", save_attention_mask=True)


        module = nn.Linear(10, 5)


        input_ids = torch.tensor([[1, 2, 3]])


        input_data = ({"input_ids": input_ids},)



        detector.process_activations(module, input_data, None)



        captured = detector.get_captured_attention_mask()


        assert captured is not None


        assert captured.shape == input_ids.shape


        assert torch.all(captured == True)




class TestModelInputDetectorGetCaptured:


    """Tests for get_captured methods."""



    def test_get_captured_input_ids_returns_tensor(self):


        """Test that get_captured_input_ids returns captured tensor."""


        detector = ModelInputDetector(layer_signature="layer_0")


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        detector.set_inputs_from_encodings({"input_ids": input_ids})



        captured = detector.get_captured_input_ids()



        assert captured is not None


        assert isinstance(captured, torch.Tensor)


        assert captured.device.type == "cpu"



    def test_get_captured_input_ids_returns_none_when_nothing_captured(self):


        """Test that get_captured_input_ids returns None when nothing captured."""


        detector = ModelInputDetector(layer_signature="layer_0")


        captured = detector.get_captured_input_ids()


        assert captured is None



    def test_get_captured_attention_mask_returns_tensor(self):


        """Test that get_captured_attention_mask returns captured tensor."""


        detector = ModelInputDetector(layer_signature="layer_0", save_attention_mask=True)


        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        attention_mask = torch.ones(2, 3)


        detector.set_inputs_from_encodings({"input_ids": input_ids, "attention_mask": attention_mask})



        captured = detector.get_captured_attention_mask()



        assert captured is not None


        assert isinstance(captured, torch.Tensor)



    def test_get_captured_attention_mask_returns_none_when_nothing_captured(self):


        """Test that get_captured_attention_mask returns None when nothing captured."""


        detector = ModelInputDetector(layer_signature="layer_0")


        captured = detector.get_captured_attention_mask()


        assert captured is None




class TestModelInputDetectorClearCaptured:


    """Tests for clear_captured method."""



    def test_clear_captured_removes_all_tensors(self):


        """Test that clear_captured removes all captured tensors."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_input_ids=True,
            save_attention_mask=True
        )


        encodings = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3)
        }


        detector.set_inputs_from_encodings(encodings)



        assert detector.get_captured_input_ids() is not None


        assert detector.get_captured_attention_mask() is not None



        detector.clear_captured()



        assert detector.get_captured_input_ids() is None


        assert detector.get_captured_attention_mask() is None


        assert 'input_ids_shape' not in detector.metadata


        assert 'attention_mask_shape' not in detector.metadata



    def test_clear_captured_when_nothing_captured(self):


        """Test that clear_captured works when nothing captured."""


        detector = ModelInputDetector(layer_signature="layer_0")



        detector.clear_captured()


        assert detector.get_captured_input_ids() is None



    def test_clear_captured_partial_data(self):


        """Test that clear_captured handles partial data correctly."""


        detector = ModelInputDetector(layer_signature="layer_0", save_input_ids=True)


        detector.set_inputs_from_encodings({"input_ids": torch.tensor([[1, 2, 3]])})



        detector.clear_captured()



        assert detector.get_captured_input_ids() is None


        assert 'input_ids_shape' not in detector.metadata




class TestModelInputDetectorSpecialTokenMask:


    """Tests for special token handling in attention mask."""



    def test_init_with_special_token_ids(self):


        """Test initialization with user-provided special token IDs."""


        special_ids = [0, 1, 2, 3]


        detector = ModelInputDetector(
            layer_signature="layer_0",
            special_token_ids=special_ids
        )


        assert detector.special_token_ids == {0, 1, 2, 3}



    def test_init_with_special_token_ids_set(self):


        """Test initialization with special token IDs as set."""


        special_ids = {0, 1, 2}


        detector = ModelInputDetector(
            layer_signature="layer_0",
            special_token_ids=special_ids
        )


        assert detector.special_token_ids == {0, 1, 2}



    def test_get_special_token_ids_from_user_provided(self):


        """Test _get_special_token_ids returns user-provided IDs."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            special_token_ids=[5, 10, 15]
        )


        module = nn.Linear(10, 5)


        special_ids = detector._get_special_token_ids(module)


        assert special_ids == {5, 10, 15}



    def test_get_special_token_ids_from_context(self):


        """Test _get_special_token_ids extracts from context."""


        from unittest.mock import MagicMock



        detector = ModelInputDetector(layer_signature="layer_0")


        mock_context = MagicMock()


        mock_context.special_token_ids = {0, 1, 2, 3}


        detector.set_context(mock_context)



        module = nn.Linear(10, 5)


        special_ids = detector._get_special_token_ids(module)



        assert special_ids == {0, 1, 2, 3}



    def test_get_special_token_ids_from_context_when_user_provided_none(self):


        """Test _get_special_token_ids falls back to context when user provided is None."""


        from unittest.mock import MagicMock



        detector = ModelInputDetector(layer_signature="layer_0", special_token_ids=None)


        mock_context = MagicMock()


        mock_context.special_token_ids = {100, 101, 102}


        detector.set_context(mock_context)



        module = nn.Linear(10, 5)


        special_ids = detector._get_special_token_ids(module)



        assert special_ids == {100, 101, 102}



    def test_get_special_token_ids_no_special_tokens(self):


        """Test _get_special_token_ids returns empty set when no special tokens found."""


        detector = ModelInputDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)


        special_ids = detector._get_special_token_ids(module)


        assert special_ids == set()



    def test_create_combined_attention_mask_with_user_ids(self):


        """Test _create_combined_attention_mask with user-provided special token IDs."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            special_token_ids=[0, 2]
        )


        module = nn.Linear(10, 5)



        input_ids = torch.tensor([[0, 1, 2, 3], [2, 4, 5, 0]])


        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)


        combined_mask = detector._create_combined_attention_mask(input_ids, attention_mask, module)



        assert combined_mask.shape == input_ids.shape


        assert combined_mask.dtype == torch.bool


        expected_mask = torch.tensor([[False, True, False, True], [False, True, True, False]])


        assert torch.equal(combined_mask, expected_mask)



    def test_create_combined_attention_mask_no_special_tokens(self):


        """Test _create_combined_attention_mask preserves attention_mask when no special tokens."""


        detector = ModelInputDetector(layer_signature="layer_0")


        module = nn.Linear(10, 5)



        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])


        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)


        combined_mask = detector._create_combined_attention_mask(input_ids, attention_mask, module)



        assert combined_mask.shape == input_ids.shape


        assert combined_mask.dtype == torch.bool


        assert torch.equal(combined_mask, attention_mask)



    def test_create_combined_attention_mask_shape(self):


        """Test that combined mask shape matches input_ids shape."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            special_token_ids=[0]
        )


        module = nn.Linear(10, 5)



        for shape in [(1, 5), (2, 10)]:


            input_ids = torch.randint(0, 100, shape)


            attention_mask = torch.ones(shape, dtype=torch.bool)


            combined_mask = detector._create_combined_attention_mask(input_ids, attention_mask, module)


            assert combined_mask.shape == input_ids.shape



    def test_create_combined_attention_mask_with_none_attention_mask(self):


        """Test _create_combined_attention_mask creates mask when attention_mask is None."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            special_token_ids=[0, 2]
        )


        module = nn.Linear(10, 5)



        input_ids = torch.tensor([[0, 1, 2, 3], [2, 4, 5, 0]])


        combined_mask = detector._create_combined_attention_mask(input_ids, None, module)



        assert combined_mask.shape == input_ids.shape


        assert combined_mask.dtype == torch.bool


        expected_mask = torch.tensor([[False, True, False, True], [False, True, True, False]])


        assert torch.equal(combined_mask, expected_mask)



    def test_create_combined_attention_mask_with_context_special_tokens(self):


        """Test _create_combined_attention_mask uses context special tokens when available."""


        from unittest.mock import MagicMock



        detector = ModelInputDetector(layer_signature="layer_0", special_token_ids=None)


        mock_context = MagicMock()


        mock_context.special_token_ids = {0, 2}


        detector.set_context(mock_context)



        module = nn.Linear(10, 5)


        input_ids = torch.tensor([[0, 1, 2, 3], [2, 4, 5, 0]])


        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)


        combined_mask = detector._create_combined_attention_mask(input_ids, attention_mask, module)



        assert combined_mask.shape == input_ids.shape


        expected_mask = torch.tensor([[False, True, False, True], [False, True, True, False]])


        assert torch.equal(combined_mask, expected_mask)



    def test_process_activations_with_combined_attention_mask(self):


        """Test process_activations creates combined attention mask excluding special tokens."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_attention_mask=True,
            special_token_ids=[0, 2]
        )


        module = nn.Linear(10, 5)


        input_ids = torch.tensor([[0, 1, 2, 3]])


        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)


        input_data = ({"input_ids": input_ids, "attention_mask": attention_mask},)



        detector.process_activations(module, input_data, None)



        captured = detector.get_captured_attention_mask()


        assert captured is not None


        assert captured.shape == input_ids.shape


        assert captured.dtype == torch.bool


        expected_mask = torch.tensor([[False, True, False, True]])


        assert torch.equal(captured, expected_mask)



    def test_set_inputs_from_encodings_with_combined_attention_mask(self):


        """Test set_inputs_from_encodings creates combined attention mask excluding special tokens."""


        detector = ModelInputDetector(
            layer_signature="layer_0",
            save_attention_mask=True,
            special_token_ids=[0, 2]
        )


        input_ids = torch.tensor([[0, 1, 2, 3], [2, 4, 5, 0]])


        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)


        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}



        detector.set_inputs_from_encodings(encodings)



        captured = detector.get_captured_attention_mask()


        assert captured is not None


        assert captured.shape == input_ids.shape


        assert captured.dtype == torch.bool


        expected_mask = torch.tensor([[False, True, False, True], [False, True, True, False]])


        assert torch.equal(captured, expected_mask)



