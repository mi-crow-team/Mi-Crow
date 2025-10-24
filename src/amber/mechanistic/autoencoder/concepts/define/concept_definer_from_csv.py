import abc

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.concepts.define.concept_definer_abc import ConceptDefiner


class ConceptDefinerFromCsv(ConceptDefiner):
    def __init__(self, concept_dictionary: ConceptDictionary, neuron_texts: list[list[NeuronText]]):
        super().__init__(concept_dictionary, neuron_texts)

    @abc.abstractmethod
    def define_concept(self, idx: int):
        pass

    def define_all_concepts(self):
        for idx in range(self.n_neurons):
            self.define_concept(idx)
