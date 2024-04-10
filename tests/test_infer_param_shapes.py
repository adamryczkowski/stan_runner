from stan_runner import infer_param_shapes


def test_infer_param_shapes_0():
    params = []
    dims, names = infer_param_shapes(params)

    assert dims == {}
    assert names == {}


def test_infer_param_shapes_1():
    params = ["mu1", "mu2", "sigma"]
    dims, names = infer_param_shapes(params)

    assert dims == {"mu1": (1,), "mu2": (1,), "sigma": (1,)}
    assert names == {"mu1": ["mu1"], "mu2": ["mu2"], "sigma": ["sigma"]}


def test_infer_param_shapes_2():
    params = ["mu1", "mu2", "sigma", "mu1"]
    # Expect exception due to duplicate parameter name
    try:
        dims, names = infer_param_shapes(params)
    except Exception as e:
        pass
    else:
        assert False


def test_infer_param_shapes_3():
    params = ["mu[1]", "mu[2]", "mu[3]", "sigma"]
    dims, names = infer_param_shapes(params)

    assert dims == {"mu": (3,), "sigma": (1,)}
    assert names == {"mu": ["mu[1]", "mu[2]", "mu[3]"], "sigma": ["sigma"]}


def test_infer_param_shapes_3b():
    params = ["mu[1]", "mu[2]", "sigma", "mu[3]"]

    # Should throw, because the runs of parameters with the same base name should not be interrupted
    try:
        dims, names = infer_param_shapes(params)
    except Exception as e:
        pass
    else:
        assert False


def test_infer_param_shapes_4():
    params = ["mu[1]", "mu[3]", "mu[2]", "sigma"]
    dims, names = infer_param_shapes(params)

    assert dims == {"mu": (3,), "sigma": (1,)}
    assert names == {"mu": ["mu[1]", "mu[3]", "mu[2]"], "sigma": ["sigma"]}


def test_infer_param_shapes_5():
    params = ["x[1,1]", "x[1,2]", "x[2,1]", "x[2,2]", "x[3,1]", "x[3,2]"]
    dims, names = infer_param_shapes(params)

    assert dims == {"x": (3, 2)}
    assert names == {"x": ["x[1,1]", "x[1,2]", "x[2,1]", "x[2,2]", "x[3,1]", "x[3,2]"]}


def test_infer_param_shapes_6():
    params = ["x[1,1]", "x[1,2]", "x[2,1]", "x[2,2]", "x[3,1]", "x[3,2]", "x[1,1]"]
    # Expect exception due to duplicate parameter name
    try:
        dims, names = infer_param_shapes(params)
    except Exception as e:
        pass
    else:
        assert False


def test_infer_param_shapes_7():
    params = ["x[1,1]", "x[1,2]", "x[1,3]", "x[2,1]", "x[2,2]", "x[2,3]"]
    dims, names = infer_param_shapes(params)

    assert dims == {"x": (2, 3)}
    assert names == {"x": ["x[1,1]", "x[1,2]", "x[1,3]", "x[2,1]", "x[2,2]", "x[2,3]"]}


def test_infer_param_shapes_8():
    params = ["x[1,1]", "x[1,2]", "x[1,3]"]
    dims, names = infer_param_shapes(params)

    assert dims == {"x": (1, 3)}
    assert names == {"x": ["x[1,1]", "x[1,2]", "x[1,3]"]}


def test_infer_param_shapes_9():
    params = ["x[2,1]"]

    # Exception due to insufficient elements in parameter name
    try:
        dims, names = infer_param_shapes(params)
    except Exception as e:
        pass
    else:
        assert False


def test_infer_param_shapes_10():
    params = ["x[1,1]", "x[1,2]", "x[1,2]", "x[2,1]", "x[2,2]", "x[2,2]"]

    # Exception due to too many elements in parameter name
    try:
        dims, names = infer_param_shapes(params)
    except Exception as e:
        pass
    else:
        assert False
