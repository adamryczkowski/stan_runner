from enum import Enum


class StanErrorType(Enum):
    NO_ERROR = 0
    SYNTAX_ERROR = 1
    COMPILE_ERROR = 2
