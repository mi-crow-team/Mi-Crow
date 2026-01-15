
"""Tests for InferenceEngine."""



import pytest


import torch


from unittest.mock import Mock, patch, MagicMock



from mi_crow.language_model.inference import InferenceEngine


from mi_crow.language_model.language_model import LanguageModel


from tests.unit.fixtures import (
    create_language_model,
    create_mock_tokenizer,
)




class TestInferenceEngine:


    """Test suite for InferenceEngine."""



    def test_init(self, mock_language_model):


        """Test initialization."""


        engine = InferenceEngine(mock_language_model)


        assert engine.lm == mock_language_model



    def test_prepare_tokenizer_kwargs_none(self, mock_language_model):


        """Test preparing tokenizer kwargs with None."""


        engine = InferenceEngine(mock_language_model)


        kwargs = engine._prepare_tokenizer_kwargs(None)



        assert kwargs["padding"] is True


        assert kwargs["truncation"] is True


        assert kwargs["return_tensors"] == "pt"



    def test_prepare_tokenizer_kwargs_with_kwargs(self, mock_language_model):


        """Test preparing tokenizer kwargs with existing kwargs."""


        engine = InferenceEngine(mock_language_model)


        kwargs = engine._prepare_tokenizer_kwargs({"max_length": 128, "padding": False})



        assert kwargs["max_length"] == 128


        assert kwargs["padding"] is False


        assert kwargs["truncation"] is True


        assert kwargs["return_tensors"] == "pt"



    def test_prepare_tokenizer_kwargs_padding_longest_with_max_length(self, mock_language_model):


        """Test that padding is set to 'longest' when padding=True and max_length is provided."""


        engine = InferenceEngine(mock_language_model)


        kwargs = engine._prepare_tokenizer_kwargs({"max_length": 1024, "padding": True})



        assert kwargs["max_length"] == 1024


        assert kwargs["padding"] == "longest"


        assert kwargs["truncation"] is True


        assert kwargs["return_tensors"] == "pt"



    def test_prepare_tokenizer_kwargs_padding_true_without_max_length(self, mock_language_model):


        """Test that padding remains True when max_length is not provided."""


        engine = InferenceEngine(mock_language_model)


        kwargs = engine._prepare_tokenizer_kwargs({"padding": True})



        assert kwargs["padding"] is True


        assert kwargs["truncation"] is True


        assert kwargs["return_tensors"] == "pt"



    def test_setup_trackers_with_tracker(self, mock_language_model):


        """Test setting up trackers when tracker exists."""


        engine = InferenceEngine(mock_language_model)


        mock_tracker = Mock()


        mock_tracker.enabled = True


        mock_language_model._input_tracker = mock_tracker



        engine._setup_trackers(["text1", "text2"])



        mock_tracker.set_current_texts.assert_called_once_with(["text1", "text2"])



    def test_setup_trackers_no_tracker(self, mock_language_model):


        """Test setting up trackers when tracker is None."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model._input_tracker = None



        engine._setup_trackers(["text1", "text2"])



    def test_setup_trackers_disabled(self, mock_language_model):


        """Test setting up trackers when tracker is disabled."""


        engine = InferenceEngine(mock_language_model)


        mock_tracker = Mock()


        mock_tracker.enabled = False


        mock_language_model._input_tracker = mock_tracker



        engine._setup_trackers(["text1", "text2"])


        mock_tracker.set_current_texts.assert_not_called()



    def test_prepare_controllers_with_controllers_true(self, mock_language_model):


        """Test preparing controllers when with_controllers is True."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.layers.get_controllers = Mock(return_value=[])



        result = engine._prepare_controllers(with_controllers=True)



        assert result == []


        mock_language_model.layers.get_controllers.assert_not_called()



    def test_prepare_controllers_with_controllers_false(self, mock_language_model):


        """Test preparing controllers when with_controllers is False."""


        engine = InferenceEngine(mock_language_model)


        controller1 = Mock()


        controller1.enabled = True


        controller2 = Mock()


        controller2.enabled = False


        controller3 = Mock()


        controller3.enabled = True



        mock_language_model.layers.get_controllers = Mock(return_value=[controller1, controller2, controller3])



        result = engine._prepare_controllers(with_controllers=False)



        assert result == [controller1, controller3]


        controller1.disable.assert_called_once()


        controller2.disable.assert_not_called()


        controller3.disable.assert_called_once()



    def test_restore_controllers(self, mock_language_model):


        """Test restoring controllers."""


        engine = InferenceEngine(mock_language_model)


        controller1 = Mock()


        controller2 = Mock()



        engine._restore_controllers([controller1, controller2])



        controller1.enable.assert_called_once()


        controller2.enable.assert_called_once()



    def test_run_model_forward_no_autocast(self, mock_language_model):


        """Test running model forward without autocast."""


        engine = InferenceEngine(mock_language_model)


        enc = {"input_ids": torch.tensor([[1, 2, 3]])}


        mock_output = Mock()


        mock_language_model.context.model = Mock(return_value=mock_output)



        with patch("torch.inference_mode"):


            output = engine._run_model_forward(enc, autocast=False, device_type="cpu", autocast_dtype=None)



        assert output == mock_output


        mock_language_model.context.model.assert_called_once_with(**enc)



    def test_run_model_forward_with_autocast_cuda(self, mock_language_model):


        """Test running model forward with autocast on CUDA."""


        engine = InferenceEngine(mock_language_model)


        enc = {"input_ids": torch.tensor([[1, 2, 3]])}


        mock_output = Mock()


        mock_language_model.context.model = Mock(return_value=mock_output)



        with patch("torch.inference_mode"):


            with patch("torch.autocast") as mock_autocast:


                mock_autocast.return_value.__enter__ = Mock()


                mock_autocast.return_value.__exit__ = Mock(return_value=None)


                output = engine._run_model_forward(enc, autocast=True, device_type="cuda", autocast_dtype=None)



        assert output == mock_output


        mock_autocast.assert_called_once_with("cuda", dtype=torch.float16)



    def test_run_model_forward_with_autocast_custom_dtype(self, mock_language_model):


        """Test running model forward with autocast and custom dtype."""


        engine = InferenceEngine(mock_language_model)


        enc = {"input_ids": torch.tensor([[1, 2, 3]])}


        mock_output = Mock()


        mock_language_model.context.model = Mock(return_value=mock_output)



        with patch("torch.inference_mode"):


            with patch("torch.autocast") as mock_autocast:


                mock_autocast.return_value.__enter__ = Mock()


                mock_autocast.return_value.__exit__ = Mock(return_value=None)


                output = engine._run_model_forward(enc, autocast=True, device_type="cuda", autocast_dtype=torch.bfloat16)



        assert output == mock_output


        mock_autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)



    def test_run_model_forward_autocast_cpu(self, mock_language_model):


        """Test running model forward with autocast on CPU (should not use autocast)."""


        engine = InferenceEngine(mock_language_model)


        enc = {"input_ids": torch.tensor([[1, 2, 3]])}


        mock_output = Mock()


        mock_language_model.context.model = Mock(return_value=mock_output)



        with patch("torch.inference_mode"):


            with patch("torch.autocast") as mock_autocast:


                output = engine._run_model_forward(enc, autocast=True, device_type="cpu", autocast_dtype=None)



        assert output == mock_output


        mock_autocast.assert_not_called()



    def test_execute_inference_success(self, mock_language_model):


        """Test executing inference successfully."""


        engine = InferenceEngine(mock_language_model)


        texts = ["text1", "text2"]



        mock_enc = {"input_ids": torch.tensor([[1, 2], [3, 4]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.context.model.parameters = Mock(return_value=iter([]))


        mock_language_model.layers.get_controllers = Mock(return_value=[])



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                output, enc = engine.execute_inference(texts)



        assert output == mock_output


        assert enc == mock_enc


        mock_language_model.tokenize.assert_called_once()


        mock_language_model.context.model.eval.assert_called_once()



    def test_execute_inference_empty_texts(self, mock_language_model):


        """Test executing inference with empty texts."""


        engine = InferenceEngine(mock_language_model)



        with pytest.raises(ValueError, match="Texts list cannot be empty"):


            engine.execute_inference([])



    def test_execute_inference_no_tokenizer(self, mock_language_model):


        """Test executing inference when tokenizer is None."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.context.tokenizer = None



        with pytest.raises(ValueError, match="Tokenizer must be initialized"):


            engine.execute_inference(["text1"])



    def test_execute_inference_with_controllers_disabled(self, mock_language_model):


        """Test executing inference with controllers disabled."""


        engine = InferenceEngine(mock_language_model)


        texts = ["text1"]



        controller = Mock()


        controller.enabled = True


        mock_language_model.layers.get_controllers = Mock(return_value=[controller])


        mock_language_model.tokenize = Mock(return_value={"input_ids": torch.tensor([[1]])})


        mock_language_model.context.model = Mock(return_value=Mock())


        mock_language_model.context.model.eval = Mock()


        mock_language_model.context.model.parameters = Mock(return_value=iter([]))



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value={"input_ids": torch.tensor([[1]])}):


            with patch("torch.inference_mode"):


                engine.execute_inference(texts, with_controllers=False)



        controller.disable.assert_called_once()


        controller.enable.assert_called_once()



    def test_extract_logits_from_output_object(self, mock_language_model):


        """Test extracting logits from output object with logits attribute."""


        engine = InferenceEngine(mock_language_model)


        mock_output = Mock()


        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])


        mock_output.logits = mock_logits



        result = engine.extract_logits(mock_output)



        assert torch.equal(result, mock_logits)



    def test_extract_logits_from_tuple(self, mock_language_model):


        """Test extracting logits from tuple output."""


        engine = InferenceEngine(mock_language_model)


        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])


        output = (mock_logits,)



        result = engine.extract_logits(output)



        assert torch.equal(result, mock_logits)



    def test_extract_logits_from_tensor(self, mock_language_model):


        """Test extracting logits from tensor output."""


        engine = InferenceEngine(mock_language_model)


        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])



        result = engine.extract_logits(mock_logits)



        assert torch.equal(result, mock_logits)



    def test_extract_logits_invalid_output(self, mock_language_model):


        """Test extracting logits from invalid output."""


        engine = InferenceEngine(mock_language_model)



        with pytest.raises(ValueError, match="Unable to extract logits"):


            engine.extract_logits("invalid")



    def test_infer_texts_no_batching(self, mock_language_model, temp_store):


        """Test infer_texts without batching."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = temp_store


        texts = ["text1", "text2"]



        mock_enc = {"input_ids": torch.tensor([[1, 2], [3, 4]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.context.model.parameters = Mock(return_value=iter([]))


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                output, enc = engine.infer_texts(texts, run_name="test_run")



        assert output == mock_output


        assert enc == mock_enc


        mock_language_model.save_detector_metadata.assert_called_once_with("test_run", 0, unified=False)



    def test_infer_texts_with_batching(self, mock_language_model, temp_store):


        """Test infer_texts with batching."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = temp_store


        texts = ["text1", "text2", "text3", "text4"]



        mock_enc1 = {"input_ids": torch.tensor([[1, 2], [3, 4]])}


        mock_enc2 = {"input_ids": torch.tensor([[5, 6], [7, 8]])}


        mock_output1 = Mock()


        mock_output2 = Mock()



        mock_language_model.tokenize = Mock(side_effect=[mock_enc1, mock_enc2])


        mock_language_model.context.model = Mock(side_effect=[mock_output1, mock_output2])


        mock_language_model.context.model.eval = Mock()


        mock_language_model.context.model.parameters = Mock(return_value=iter([]))


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", side_effect=[mock_enc1, mock_enc2]):


            with patch("torch.inference_mode"):


                outputs, encodings = engine.infer_texts(texts, run_name="test_run", batch_size=2)



        assert len(outputs) == 2


        assert len(encodings) == 2


        assert mock_language_model.save_detector_metadata.call_count == 2



    def test_infer_texts_no_run_name(self, mock_language_model):


        """Test infer_texts without run_name (no metadata saving)."""


        engine = InferenceEngine(mock_language_model)


        texts = ["text1", "text2"]



        mock_enc = {"input_ids": torch.tensor([[1, 2], [3, 4]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.context.model.parameters = Mock(return_value=iter([]))


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                output, enc = engine.infer_texts(texts)



        assert output == mock_output


        assert enc == mock_enc


        mock_language_model.save_detector_metadata.assert_not_called()



    def test_infer_dataset(self, mock_language_model, temp_store):


        """Test infer_dataset."""


        from datasets import Dataset


        from mi_crow.datasets import TextDataset



        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = temp_store



        hf_dataset = Dataset.from_dict({"text": ["text1", "text2", "text3"]})


        dataset = TextDataset(hf_dataset, temp_store)



        mock_enc = {"input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.context.model.parameters = Mock(return_value=iter([]))


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                run_name = engine.infer_dataset(dataset, run_name="test_run", batch_size=2)



        assert run_name == "test_run"


        assert mock_language_model.save_detector_metadata.call_count == 2



    def test_infer_texts_uses_save_in_batches_flag(self, mock_language_model, temp_store):


        """infer_texts should propagate save_in_batches to LanguageModel."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = temp_store


        texts = ["text"]



        mock_enc = {"input_ids": torch.tensor([[1, 2]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                engine.infer_texts(texts, run_name="run_unified", save_in_batches=False)



        mock_language_model.save_detector_metadata.assert_called_once_with("run_unified", 0, unified=True)



    def test_infer_texts_clears_detectors_when_requested(self, mock_language_model, temp_store):


        """infer_texts should clear detectors when clear_detectors_before is True."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = temp_store



        texts = ["text1", "text2"]


        mock_enc = {"input_ids": torch.tensor([[1, 2], [3, 4]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()


        mock_language_model.clear_detectors = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                engine.infer_texts(
                    texts,
                    run_name="test_run",
                    clear_detectors_before=True,
                )



        mock_language_model.clear_detectors.assert_called_once()



    def test_infer_dataset_clears_detectors_when_requested(self, mock_language_model, temp_store):


        """infer_dataset should clear detectors when clear_detectors_before is True."""


        from datasets import Dataset


        from mi_crow.datasets import TextDataset



        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = temp_store



        hf_dataset = Dataset.from_dict({"text": ["text1", "text2", "text3"]})


        dataset = TextDataset(hf_dataset, temp_store)



        mock_enc = {"input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()


        mock_language_model.clear_detectors = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                engine.infer_dataset(
                    dataset,
                    run_name="test_run",
                    batch_size=2,
                    clear_detectors_before=True,
                )



        mock_language_model.clear_detectors.assert_called_once()



    def test_extract_dataset_info_happy_and_fallback(self, mock_language_model):


        """_extract_dataset_info should return dataset metadata or a safe fallback."""


        engine = InferenceEngine(mock_language_model)



        class GoodDataset:


            def __init__(self):


                self.dataset_dir = "/tmp/ds"



            def __len__(self):


                return 5



        class BadDataset:


            dataset_dir = "/tmp/bad"



            def __len__(self):


                raise RuntimeError("boom")



        good = GoodDataset()


        bad = BadDataset()



        info_good = engine._extract_dataset_info(good)


        assert info_good == {"dataset_dir": "/tmp/ds", "length": 5}



        info_bad = engine._extract_dataset_info(bad)


        assert info_bad == {"dataset_dir": "", "length": -1}



    def test_prepare_run_metadata_includes_dataset_and_options(self, mock_language_model):


        """_prepare_run_metadata should include dataset info and options when provided."""


        engine = InferenceEngine(mock_language_model)



        class DatasetStub:


            def __init__(self):


                self.dataset_dir = "/ds"



            def __len__(self):


                return 10



        dataset = DatasetStub()


        options = {"batch_size": 4, "max_length": 32}



        run_name, meta = engine._prepare_run_metadata(dataset=dataset, run_name="run-1", options=options)



        assert run_name == "run-1"


        assert meta["run_name"] == "run-1"


        assert "model" in meta


        assert meta["options"] == options


        assert meta["dataset"] == {"dataset_dir": "/ds", "length": 10}




        auto_name, auto_meta = engine._prepare_run_metadata(dataset=None, run_name=None, options=None)


        assert isinstance(auto_name, str) and auto_name


        assert auto_meta["run_name"] == auto_name



    def test_save_run_metadata_handles_store_errors(self, mock_language_model):


        """_save_run_metadata should swallow store errors and log when verbose=True."""


        engine = InferenceEngine(mock_language_model)



        class FailingStore:


            def put_run_metadata(self, run_name, meta):


                raise OSError("disk full")



        store = FailingStore()


        meta = {"run_name": "x"}



        with patch("mi_crow.language_model.inference.logger") as mock_logger:



            engine._save_run_metadata(store, "x", meta, verbose=True)



        assert mock_logger.warning.called



    def test_infer_texts_with_run_name_and_no_store_raises(self, mock_language_model):


        """infer_texts should require a store when run_name is provided."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = None



        with pytest.raises(ValueError, match="Store must be provided to save metadata"):


            engine.infer_texts(["text"], run_name="run-1")



    def test_infer_dataset_requires_model_and_store(self, mock_language_model, temp_store):


        """infer_dataset should validate that model and store are present."""


        engine = InferenceEngine(mock_language_model)



        class DatasetStub:


            def iter_batches(self, batch_size):


                yield ["x"]



            def extract_texts_from_batch(self, batch):


                return batch



        ds = DatasetStub()




        mock_language_model.context.model = None


        mock_language_model.store = temp_store


        with pytest.raises(ValueError, match="Model must be initialized before running"):


            engine.infer_dataset(ds, run_name="r")




        mock_language_model.context.model = Mock()


        mock_language_model.store = None


        with pytest.raises(ValueError, match="Store must be provided or set on the language model"):


            engine.infer_dataset(ds, run_name="r")



    def test_infer_dataset_skips_empty_batches(self, mock_language_model, temp_store):


        """infer_dataset should gracefully skip empty batches from the dataset."""


        engine = InferenceEngine(mock_language_model)


        mock_language_model.store = temp_store



        class DatasetWithEmpty:


            dataset_dir = "ds"



            def __len__(self):


                return 2



            def iter_batches(self, batch_size):


                yield []


                yield ["a", "b"]



            def extract_texts_from_batch(self, batch):


                return batch



        ds = DatasetWithEmpty()



        mock_enc = {"input_ids": torch.tensor([[1, 2]])}


        mock_output = Mock()



        mock_language_model.tokenize = Mock(return_value=mock_enc)


        mock_language_model.context.model = Mock(return_value=mock_output)


        mock_language_model.context.model.eval = Mock()


        mock_language_model.context.model.parameters = Mock(return_value=iter([]))


        mock_language_model.layers.get_controllers = Mock(return_value=[])


        mock_language_model.save_detector_metadata = Mock()



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=mock_enc):


            with patch("torch.inference_mode"):


                run_name = engine.infer_dataset(ds, run_name="run-empty", batch_size=2)



        assert run_name == "run-empty"



        mock_language_model.save_detector_metadata.assert_called_once()



    def test_infer_texts_stop_after_layer_changes_output_shape(self, temp_store):


        """infer_texts with stop_after_layer should return an intermediate layer output."""


        from tests.unit.fixtures.language_models import create_language_model_from_mock




        lm = create_language_model_from_mock(temp_store)


        engine = InferenceEngine(lm)



        texts = ["hello world"]


        enc = {"input_ids": torch.tensor([[1, 2, 3]])}




        lm.tokenize = Mock(return_value=enc)



        with patch("mi_crow.language_model.inference.move_tensors_to_device", return_value=enc):



            full_output, _ = engine.infer_texts(texts)



            early_output, _ = engine.infer_texts(texts, stop_after_layer=0)



        assert isinstance(full_output, torch.Tensor)


        assert isinstance(early_output, torch.Tensor)




        assert early_output.shape[0] == full_output.shape[0]


        assert early_output.shape[1] == full_output.shape[1]




        assert early_output.shape[-1] != full_output.shape[-1]



