from abc import ABC, abstractmethod

generators_dict = {}


class Generator(ABC):
    """
    Abstract class able to provide a way to standardize the construction of a generator. Generator is a callable object
    able to provide a way to run heavy analysis and store results in a json file.
    """

    def __init__(self, key: str):
        """
        Constructor of generator.
        @param key: Key of generator able to retrieve it from a central dictionary.
        """
        generators_dict[key] = self

    @abstractmethod
    def __call__(self, file_to_save: str):
        """
        Abstract method able to provide a way to run the heavy analysis and store results in a json file.
        @param file_to_save: File to store analysis results.
        """
        pass
