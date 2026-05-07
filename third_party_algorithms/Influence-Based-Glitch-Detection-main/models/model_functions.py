from abc import ABC, abstractmethod


class ModelFunctions(ABC):
    @abstractmethod
    def is_pretrained(self):
        raise NotImplementedError("is_pretrained method must be implemented")

    @abstractmethod
    def trainable_layer_names(self):
        return None

    @abstractmethod
    def get_model_instance(self):
        raise NotImplementedError("get_model_instance method must be implemented")
