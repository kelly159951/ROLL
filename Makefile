.PHONY: test precommit benchmark_core benchmark_aux

check_dirs := examples tests sppo

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

precommit:
	pre-commit run --all-files

benchmark_core:
	bash ./benchmark/benchmark_core.sh

benchmark_aux:
	bash ./benchmark/benchmark_aux.sh


qwen3-1.7B-rlvr_megatron:
	export PYTHONPATH="${PYTHONPATH}:$(pwd)" ;\
	export WANDB_MODE=offline ;\
	bash examples/qwen3-1.7B-rlvr_megatron/run_rlvr_pipeline.sh