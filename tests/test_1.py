import numpy as np

from stan_runner import *
from pathlib import Path
import numpy as np


def model_generator_array(total_dim: list[int],
                          matrix_size: int = 0,
                          incl_transformed_params: bool = False) -> tuple[str, dict[str, np.ndarray]]:
    # Returns stan model that gets input data of dimension data_dim, and returns something
    # of the same dimension.

    assert len(total_dim) >= matrix_size
    assert matrix_size <= 2
    array_size = len(total_dim) - matrix_size
    matrix_dim = total_dim[-matrix_size:]
    array_dim = total_dim[:-matrix_size]

    if array_size > 0:
        array_str = "array [" + "][".join([str(i) for i in array_dim]) + "] "
    else:
        array_str = ""

    if matrix_size == 0 or (matrix_size == 1 and matrix_dim[0] == 1):
        matrix_type = "real"
    elif matrix_size == 1:
        matrix_type = f"vector[{matrix_dim[0]}]"
    elif matrix_size == 2:
        matrix_type = f"matrix[{matrix_dim[0]}, {matrix_dim[1]}]"
    else:
        raise Exception("matrix_size should be 0, 1, or 2.")

    if incl_transformed_params:
        transformed_params_str = f"""
generated quantities {{
    {array_str}{matrix_type} arr2;
    arr2 = arr + 1;
}}
"""
    else:
        transformed_params_str = ""

    if total_dim == [1]:
        data_str = """
data {
    int dims;
"""
    else:
        data_str = f"""
data {{
    int N;
    int dims[N];
"""
    model_str = f"""
{data_str}
    {array_str}{matrix_type} arr;
}}
parameters {{
    {array_str}{matrix_type} par;
}}
model {{
    par ~ normal(0, 1);
    arr ~ normal(par, 1);
}}
{transformed_params_str}

"""
    arr = np.random.randn(*total_dim)
    if total_dim == [1]:
        data_dict = {"dims": 1, "arr": arr.tolist()[0]}
    else:
        data_dict = {"N": np.array(total_dim, dtype=int),
                     "dims": np.array(total_dim, dtype=int),
                     "arr": arr}

    return model_str, data_dict


def test1():
    # model_cache_dir should be an absolute string of the model_cache directory in the project's folder.
    model_cache_dir = Path(__file__).parent / "model_cache"

    runner = RPyRunner(model_cache=model_cache_dir)
    runner.install_dependencies()

    model_code, data = model_generator_array([1], 1, True)
    runner.load_model_by_str(model_code, "test_model")
    if not runner.is_model_loaded:
        print(runner.messages["stanc_error"])
        return
    runner.compile_model()
    if not runner.is_model_compiled:
        print(runner.messages["compile_error"])
        return
    runner.set_data(data)
    if not runner.is_data_set:
        print(runner.messages["data_error"])
        return
    runner.sampling(num_samples=1000, num_chains=8)
    print(runner)


if __name__ == '__main__':
    test1()
