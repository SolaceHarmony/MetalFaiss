from enum import Enum
from typing import Optional

class QuantizerType(Enum):
    """Available quantizer types for scalar quantization"""
    
    QT8BIT = 0
    QT4BIT = 1
    QT8BIT_UNIFORM = 2
    QT4BIT_UNIFORM = 3
    QT_FP16 = 4
    QT8BIT_DIRECT = 5
    QT6BIT = 6
    
    @classmethod
    def from_string(cls, name: str) -> 'QuantizerType':
        """
        Convert string representation to QuantizerType.
        
        Args:
            name: String name of quantizer type (case insensitive)
            
        Returns:
            Corresponding QuantizerType
            
        Raises:
            ValueError: If quantizer type name is not recognized
        """
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown quantizer type: {name}")
    
    def __str__(self) -> str:
        """Return lowercase string representation of quantizer type."""
        return self.name.lower()
