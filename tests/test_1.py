from pathlib import Path

import numpy as np

from stan_runner import *


def model_generator_array(total_dim: list[int],
                          matrix_dim_count: int = 0,
                          incl_transformed_params: bool = False) -> tuple[str, dict[str, np.ndarray]]:
    # Returns stan model that gets input data of dimension data_dim, and returns something
    # of the same dimension.

    assert len(total_dim) >= matrix_dim_count
    assert matrix_dim_count <= 2
    total_dim_count = len(total_dim)
    array_dim_count = total_dim_count - matrix_dim_count
    if matrix_dim_count > 0:
        matrix_dim = total_dim[-matrix_dim_count:]
        array_dim = total_dim[:-matrix_dim_count]
    else:
        matrix_dim = []
        array_dim = total_dim

    if array_dim_count > 0:
        array_str = "array [" + ",".join([str(i) for i in array_dim]) + "] "
    else:
        array_str = ""

    if matrix_dim_count == 0 or (matrix_dim_count == 1 and matrix_dim[0] == 1):
        matrix_type = "real"
    elif matrix_dim_count == 1:
        matrix_type = f"vector[{matrix_dim[0]}]"
    elif matrix_dim_count == 2:
        matrix_type = f"matrix[{matrix_dim[0]}, {matrix_dim[1]}]"
    else:
        raise Exception("matrix_size should be 0, 1, or 2.")

    model_code = \
        [f"data {{",
         f"   {array_str}{matrix_type} arr;",
         f"}}",
         "",
         f"parameters {{",
         f"   {array_str}{matrix_type} par;",
         f"}}",
         "",
         f"model {{",
         ]
    indent = ""
    arr_index = ""
    braces_off = ""
    for a in range(array_dim_count):
        # Sets the identifier for the array dimension as i, j, k, l, and so on for each array_size
        identifier = chr(ord('i') + a)
        indent = "   " * a
        model_code.append(f"   {indent}for ({identifier} in 1:{array_dim[a]}) {{")
        arr_index += f"[{identifier}]"
        braces_off += "}"

    if matrix_dim_count == 2:
        indent += "   "
        braces_off += "}"
        model_code.append(f"   {indent}for (row in 1:{matrix_dim[0]}) {{")
        model_code.append(f"   {indent}   arr{arr_index}[row] ~ normal(par{arr_index}[row], 1);")
    else:
        model_code.append(f"   {indent}arr{arr_index} ~ normal(par{arr_index}, 1);")
    model_code.append(f"   {braces_off}")
    model_code.append("}")
    model_code.append("")
    model_code.append("generated quantities {")
    model_code.append(f"   {array_str}{matrix_type} par2;")
    if array_dim_count > 0:
        arr_index = ""
        braces_off = ""
        for a in range(array_dim_count):
            identifier = chr(ord('i') + a)
            indent = "   " * a
            model_code.append(f"   {indent}for ({identifier} in 1:{array_dim[a]}) {{")
            arr_index += f"[{identifier}]"
            braces_off += "}"
        model_code.append(f"   {indent}par2{arr_index} = par{arr_index} + 1;")
        model_code.append(f"   {braces_off}")
    else:
        model_code.append("   par2 = par + 1;")
    model_code.append("}")
    model_code.append("")
    model_str = "\n".join(model_code)

    arr = np.random.randn(*total_dim)
    if total_dim == [1]:
        data_dict = {"dims": 1, "arr": arr.tolist()[0]}
    else:
        data_dict = {"N": np.array(total_dim, dtype=int),
                     "dims": np.array(total_dim, dtype=int),
                     "arr": arr}

    return model_str, data_dict


def test_model(model_code: str, data: dict[str, str]):
    # model_cache_dir should be an absolute string of the model_cache directory in the project's folder.
    model_cache_dir = Path(__file__).parent / "model_cache"

    runner = RPyRunner(model_cache=model_cache_dir)
    runner.install_dependencies()

    runner.load_model_by_str(model_code, "test_model")
    if not runner.is_model_loaded:
        print(runner.model_code)
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
    if not runner.is_model_sampled:
        print(runner.messages["sampling_error"])
        return
    print(runner)


def test1():
    test_model(*model_generator_array([1], 1, True))


def test2():
    test_model(*model_generator_array([2], 1, True))


def test3():
    test_model(*model_generator_array([2, 3], 2, True))


def test4():
    test_model(*model_generator_array([2, 3, 4], 2, True))


def test4():
    test_model(*model_generator_array([2, 3, 4, 2], 2, True))


def test5():
    test_model(*model_generator_array([2, 3, 4], 1, True))


def test6():
    test_model(*model_generator_array([2, 3, 4], 0, True))


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
