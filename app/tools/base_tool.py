from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Base class for all tools.

    Every concrete tool should define:
    - name: stable unique identifier for registry lookup
    - description: short explanation of tool capability
    - run(input): execute tool logic and return output text
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, input: str) -> str:
        """Execute tool logic with natural-language input."""
        raise NotImplementedError
