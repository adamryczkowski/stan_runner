import numpy as np
import re
from typing import Any


def rcode_load_library(library_name: str) -> str:
    out = (
            'if(! "'
            + library_name
            + '" %in% installed.packages()) {install.packages("'
            + library_name
            + '" )}'
    )
    return out


def normalize_stan_code(code: str) -> str:
    # R code:
    # normalize_code_from_lines <- function(lines) {
    #   lines <- lines[which(nchar(lines) > 0)[1]:length(lines)]
    #   # Replace every occurence of an empty line white space with a just a an empty line
    #   lines <- gsub("^\\s*$", "", lines)
    #   model_string <- paste0(lines, collapse = "\n")
    #   file_string <- gsub("\n{2,}", "\n", model_string)
    #   # Add empty line at the end
    #   file_string <- paste0(file_string, "\n")
    #   return(file_string)
    # }

    lines = code.split("\n")
    lines = [line for line in lines if len(line.strip()) > 0]
    model_string = "\n".join(lines)
    file_string = re.sub("\n{2,}", "\n", model_string)
    file_string = file_string + "\n"
    return file_string


def convert_dict_to_r(d: dict[str, np.ndarray | float | int]) -> Any:
    """Convert a dictionary to R object"""
    from rpy2.robjects.packages import importr
    base = importr("base")
    d_r = base.list()
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()
    for key, item in d.items():
        if isinstance(type, np.ndarray):
            if item.shape == ():
                item = item.tolist()
        d_r.rx2[key] = item
    return d_r
