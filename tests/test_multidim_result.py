from pathlib import Path

import numpy as np

from stan_runner import *


def model_generator_array(main: StanBackend, total_dim: list[int],
                          matrix_dim_count: int = 0,
                          incl_transformed_params: bool = False) -> tuple[IStanModel, IStanData]:
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
    model_code.append(f"   {array_str}{matrix_type} par3;")
    if array_dim_count > 0:
        arr_index = ""
        braces_off = ""
        for a in range(array_dim_count):
            identifier = chr(ord('i') + a)
            indent = "   " * a
            model_code.append(f"   {indent}for ({identifier} in 1:{array_dim[a]}) {{")
            arr_index += f"[{identifier}]"
            braces_off += "}"
        model_code.append(f"   {indent}par3{arr_index} = par{arr_index} + 1;")
        model_code.append(f"   {braces_off}")
    else:
        model_code.append("   par3 = par + 1;")
    model_code.append("}")
    model_code.append("")
    model_str = "\n".join(model_code)

    model_name = f"test_dims_{'-'.join([str(x) for x in total_dim])}_matrix_{matrix_dim_count}"

    model_obj = main.make_model(model_name=model_name, model_code=model_str)

    arr = np.random.randn(*total_dim)
    if total_dim == [1]:
        data_dict = {"dims": 1, "arr": arr.tolist()[0]}
    else:
        data_dict = {"N": np.array(total_dim, dtype=int),
                     "dims": np.array(total_dim, dtype=int),
                     "arr": arr}

    data_obj = main.make_data(data=data_dict)

    return model_obj, data_obj


def test_model(model_obj: StanModel, data_obj: StanData, output_scope: StanOutputScope = StanOutputScope.MainEffects,
               run_engine: StanResultEngine = StanResultEngine.MCMC):
    # model_cache_dir should be an absolute string of the model_cache directory in the project's folder.
    # model_cache_dir = Path(__file__).parent.parent / "model_cache"
    install_all_dependencies()
    # runner = CmdStanRunner(model_cache=model_cache_dir)
    # runner.install_dependencies()
    try:
        model_obj.make_sure_is_compiled()
    except ValueError as v:
        print(f"Compilation error: {str(v)}")
        return

    assert model_obj.is_compiled

    runner = StanRun(run_folder=Path(__file__).parent.parent / "run_cache",
                     data=data_obj, model=model_obj, output_scope=output_scope,
                     run_engine=run_engine, run_opts={"sample_count": 4000})

    result = runner.run()
    # print(messages)
    print(result)


def test1(main: StanBackend):
    test_model(*model_generator_array(main, [1], 1, True))


def test2(main: StanBackend):
    test_model(*model_generator_array(main, [2], 1, True))


def test3(main: StanBackend):
    test_model(*model_generator_array(main, [2, 3], 2, True))


def test4(main: StanBackend):
    test_model(*model_generator_array(main, [2, 3, 4], 2, True))


def test4(main: StanBackend):
    test_model(*model_generator_array(main, [2, 3, 4, 2], 2, True))


def test5(main: StanBackend):
    test_model(*model_generator_array(main, [2, 3, 4], 1, True))


def test6(main: StanBackend):
    test_model(*model_generator_array(main, [2, 3, 4], 0, True))


if __name__ == '__main__':
    main = StanBackend()
    test1(main)
    test2(main)
    test3(main)
    test4(main)
    test5(main)
    test6(main)
