from abc import ABC, abstractmethod
import mlx.core as mx
import math

###############################################################################
# Abstract Base Classes
###############################################################################

class DistanceComputer(ABC):
    """
    Base class for computing distances from a current query to stored vectors.
    """

    @abstractmethod
    def set_query(self, x: mx.array) -> None:
        """Set the current query vector. x must be an MLX array."""
        pass

    @abstractmethod
    def __call__(self, i: int) -> float:
        """Compute the distance from the query to the stored vector at index i."""
        pass

    def distances_batch_4(self, idx0: int, idx1: int, idx2: int, idx3: int) -> tuple:
        """Compute distances to four stored vectors; default implementation calls __call__ four times."""
        d0 = self(idx0)
        d1 = self(idx1)
        d2 = self(idx2)
        d3 = self(idx3)
        return (d0, d1, d2, d3)

    @abstractmethod
    def symmetric_dis(self, i: int, j: int) -> float:
        """Compute the distance between stored vectors at indices i and j."""
        pass


class NegativeDistanceComputer(DistanceComputer):
    """
    A wrapper around a DistanceComputer that negates the computed distances.
    Useful for inner-product (similarity) search where maximizing is equivalent
    to minimizing the negative score.
    """
    def __init__(self, base_computer: DistanceComputer):
        self.base_computer = base_computer

    def set_query(self, x: mx.array) -> None:
        self.base_computer.set_query(x)

    def __call__(self, i: int) -> float:
        return -self.base_computer(i)

    def distances_batch_4(self, idx0: int, idx1: int, idx2: int, idx3: int) -> tuple:
        d0, d1, d2, d3 = self.base_computer.distances_batch_4(idx0, idx1, idx2, idx3)
        return (-d0, -d1, -d2, -d3)

    def symmetric_dis(self, i: int, j: int) -> float:
        return -self.base_computer.symmetric_dis(i, j)


###############################################################################
# Flat Codes Distance Computer
###############################################################################

class FlatCodesDistanceComputer(DistanceComputer):
    """
    Base class for distance computers that operate on a flat layout of encoded
    vectors. The encoded data is stored in an MLX array.
    """
    def __init__(self, codes: mx.array, code_size: int):
        self.codes = codes  # MLX array holding the codes
        self.code_size = code_size
        self.query = None  # to be set via set_query()

    def set_query(self, x: mx.array) -> None:
        self.query = x

    def __call__(self, i: int) -> float:
        # Assuming that codes[i] returns an MLX array slice of length code_size
        code = self.codes[i]
        return self.distance_to_code(code)

    @abstractmethod
    def distance_to_code(self, code: mx.array) -> float:
        """
        Compute the distance between the current query (set via set_query)
        and the encoded vector stored in 'code'.
        """
        pass

    @abstractmethod
    def symmetric_dis(self, i: int, j: int) -> float:
        """Compute the distance between stored vectors at indices i and j."""
        pass


###############################################################################
# Concrete Implementations: Squared L2 and Inner Product
###############################################################################

class ExtraL2DistanceComputer(FlatCodesDistanceComputer):
    """
    A concrete implementation of FlatCodesDistanceComputer for squared L2 distance.
    """
    def __init__(self, codes: mx.array, nb: int, d: int):
        super().__init__(codes, d)
        self.nb = nb  # number of stored vectors

    def distance_to_code(self, code: mx.array) -> float:
        if self.query is None:
            raise ValueError("Query not set in ExtraL2DistanceComputer")
        diff = self.query - code  # MLX array subtraction
        # Use MLX dot-product; this returns an MLX scalar
        return float(mx.dot(diff, diff))

    def symmetric_dis(self, i: int, j: int) -> float:
        diff = self.codes[i] - self.codes[j]
        return float(mx.dot(diff, diff))


class ExtraIPDistanceComputer(FlatCodesDistanceComputer):
    """
    A concrete implementation of FlatCodesDistanceComputer for inner product.
    """
    def __init__(self, codes: mx.array, nb: int, d: int):
        super().__init__(codes, d)
        self.nb = nb

    def distance_to_code(self, code: mx.array) -> float:
        if self.query is None:
            raise ValueError("Query not set in ExtraIPDistanceComputer")
        return float(mx.dot(self.query, code))

    def symmetric_dis(self, i: int, j: int) -> float:
        return float(mx.dot(self.codes[i], self.codes[j]))


###############################################################################
# Factory Function
###############################################################################

def get_extra_distance_computer(d: int, mt: str, metric_arg: float, nb: int, xb: mx.array) -> DistanceComputer:
    """
    Factory function that returns an instance of a DistanceComputer based on the
    metric type. The metric type is a string (e.g. "L2" or "IP").
    
    Args:
        d: The dimension of the vectors.
        mt: The metric type, e.g., "L2" for squared L2 distance or "IP" for inner product.
        metric_arg: (Unused here, but may be used for specialized metrics.)
        nb: The number of database vectors.
        xb: The database vectors stored as an MLX array.
    
    Returns:
        An instance of DistanceComputer.
    """
    if mt.upper() == "L2":
        return ExtraL2DistanceComputer(xb, nb, d)
    elif mt.upper() == "IP":
        return ExtraIPDistanceComputer(xb, nb, d)
    else:
        raise ValueError(f"Unsupported metric type: {mt}")