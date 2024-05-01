from .cmdstan_runner import CmdStanRunner
from overrides import overrides
from .ifaces import ILocalInferenceResult
from typing import Any

class CachedCmdStanRunner(CmdStanRunner):

    def
    @overrides
    def sampling(self, num_chains: int, iter_sampling: int = None,
                 iter_warmup: int = None, thin: int = 1, max_treedepth: int = None,
                 seed: int = None, inits: dict[str, Any] | float | list[str] = None) -> ILocalInferenceResult:

