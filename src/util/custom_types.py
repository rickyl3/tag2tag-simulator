from types import UnionType

type Position = tuple[float, float, float]

type StateMethod = UnionType[tuple[StateMethod, ...], str, int]
