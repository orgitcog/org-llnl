from abc import ABCMeta, abstractmethod

class Method(object):
    """
    Abstract class representing a generic Method.
    Attributes:
        name (str): name of the methods (e.g., BayesMTL)
        paradigm (str): denote if the method is MTL (Multitask Learning model) or 
                        STL (Singletask Learning model)
        mode (bool): denote whether the method is in train model or evaluation mode

    """
    __metaclass__ = ABCMeta

    def __init__(self, name, paradigm): #, full_name):
        """
        Class initialization.
        Args
            name (str): name to be used as reference
        """
        self.name = name
        self.paradigm = paradigm
        self.mode = None  # train / test

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self):
        """
        Train method's parameters.
        Args
            name (np.array()):
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Perform prediction.
        Args
            name (np.array()):
        Return
            name (np.array()):
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Perform prediction.
        Args
            name (np.array()):
        Return
            name (np.array()):
        """
        pass

    @abstractmethod
    def hyperparameters_selection(self):
        """
        Systematically performs method's hyper-parameters selection.
        Args
            name (np.array()):
        Return
            name (np.array()):
        """
        pass

    @abstractmethod
    def load_params(self):
        """
        Load method's parameters.
        Args
            name (np.array()):
        """
        pass

    @abstractmethod
    def set_params(self):
        """
        Set method's parameters.
        Args
            name (np.array()):
        """
        pass

    @abstractmethod
    def feature_importance(self):
        """
        Extract feature importance from trained model.
        """
        pass

    def get_hyperparameter_space(self):
        """
        Set method's hyper-parameters space.
        Args
            name (np.array()):
        """
        if self.hyperparameter_space is None:
            raise ValueError('Hyperparameter space not defined yet.')
        return self.hyperparameter_space

    def set_mode(self, mode):
        self.mode = mode