from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the AMMFS system.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Main processing method for the agent.
        """
        pass

    def __repr__(self):
        return f"<Agent: {self.name}>"
