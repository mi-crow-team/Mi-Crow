import abc

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText


class ConceptDefiner(abc.ABC):
    def __init__(
            self,
            concept_dictionary: ConceptDictionary,
            neuron_texts: list[list[NeuronText]]
    ):
        self.concept_dictionary = concept_dictionary
        self.neuron_texts = neuron_texts

        self.n_neurons = concept_dictionary.n_size

    @abc.abstractmethod
    def define_concept(self, idx: int):
        pass

    def define_all_concepts(self):
        for idx in range(self.n_neurons):
            self.define_concept(idx)
