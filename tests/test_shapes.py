def test_imports():
    import saliency_bench.core.runner as runner

    assert hasattr(runner, "run_experiment")

