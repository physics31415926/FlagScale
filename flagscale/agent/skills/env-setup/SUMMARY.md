# Env-Setup — Summary

Set up FlagScale training environment on GPU servers with all FL-customized dependencies.

**Load when**: creating a new conda environment, installing FlagScale dependencies, resolving CUDA/PyTorch version conflicts, or debugging import errors.

Strategy: collect ALL constraints first (driver, framework, recipe), solve for compatible versions, then one-shot install. Mandatory deps: Megatron-LM-FL, TransformerEngine-FL, Apex (with CUDA extensions), Flash-Attention. Always use `--no-deps` for packages that pull PyTorch.
