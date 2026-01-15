
import pytest


from datasets import Dataset



from mi_crow.datasets.base_dataset import BaseDataset


from mi_crow.datasets.loading_strategy import LoadingStrategy


from tests.unit.fixtures.stores import create_temp_store




class DummyDataset(BaseDataset):


    def __init__(self, data, store, strategy=LoadingStrategy.MEMORY):


        ds = Dataset.from_dict({"text": data})


        super().__init__(ds, store=store, loading_strategy=strategy)



    def __len__(self):


        return len(self._ds)



    def __getitem__(self, idx):


        if isinstance(idx, slice):


            return [self._ds[i]["text"] for i in range(*idx.indices(len(self)))]


        if isinstance(idx, list):


            return [self._ds[i]["text"] for i in idx]


        return self._ds[idx]["text"]



    def iter_items(self):


        for item in self._ds["text"]:


            yield item



    def iter_batches(self, batch_size):


        batch = []


        for item in self.iter_items():


            batch.append(item)


            if len(batch) == batch_size:


                yield batch


                batch = []


        if batch:


            yield batch



    def extract_texts_from_batch(self, batch):


        return list(batch)



    def get_all_texts(self):


        return list(self._ds["text"])




def test_base_dataset_batching_head_and_sample(tmp_path):


    store = create_temp_store(tmp_path)


    data = [f"item-{i}" for i in range(5)]


    ds = DummyDataset(data, store)



    assert ds.get_batch(1, 2) == ["item-1", "item-2"]


    assert ds.head(2) == ["item-0", "item-1"]


    sampled = ds.sample(3)


    assert len(sampled) == 3


    assert all(item in data for item in sampled)


    assert len(set(sampled)) == 3


    samples = [tuple(sorted(ds.sample(3))) for _ in range(5)]


    unique_samples = set(samples)


    assert len(unique_samples) >= 2




def test_base_dataset_sample_iterable_only(tmp_path):


    store = create_temp_store(tmp_path)


    data = ["a", "b"]


    iterable_ds = DummyDataset(data, store, strategy=LoadingStrategy.STREAMING)


    with pytest.raises(NotImplementedError):


        iterable_ds.sample(1)




def test_base_dataset_is_streaming_flags(tmp_path):


    store = create_temp_store(tmp_path)


    memory_ds = DummyDataset(["a"], store, strategy=LoadingStrategy.MEMORY)


    dynamic_ds = DummyDataset(["a"], store, strategy=LoadingStrategy.DISK)


    iterable_ds = DummyDataset(["a"], store, strategy=LoadingStrategy.STREAMING)



    assert memory_ds.is_streaming is False


    assert dynamic_ds.is_streaming is True


    assert iterable_ds.is_streaming is True




def test_base_dataset_invalid_strategy(tmp_path):


    store = create_temp_store(tmp_path)


    ds = Dataset.from_dict({"text": ["a"]})



    class InvalidDataset(BaseDataset):


        def __init__(self):


            super().__init__(ds, store, loading_strategy="invalid")



        def __len__(self):


            return 0



        def __getitem__(self, idx):


            return None



        def iter_items(self):


            yield from ()



        def iter_batches(self, batch_size):


            yield from ()



        def extract_texts_from_batch(self, batch):


            return []



        def get_all_texts(self):


            return []



    with pytest.raises(ValueError):


        InvalidDataset()




def test_base_dataset_without_store_directory(tmp_path):


    class StoreStub:


        base_path = ""


        dataset_prefix = "datasets"



    ds = Dataset.from_dict({"text": ["a"]})



    class MinimalDataset(BaseDataset):


        def __init__(self):


            super().__init__(ds, store=StoreStub(), loading_strategy=LoadingStrategy.MEMORY)



        def __len__(self):


            return 1



        def __getitem__(self, idx):


            if isinstance(idx, slice):


                return ["a"]


            return "a"



        def iter_items(self):


            yield "a"



        def iter_batches(self, batch_size):


            yield ["a"]



        def extract_texts_from_batch(self, batch):


            return list(batch)



        def get_all_texts(self):


            return ["a"]



    dataset = MinimalDataset()


    assert dataset.head(1) == ["a"]


