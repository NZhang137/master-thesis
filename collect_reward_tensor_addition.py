# --- append to src/proxy_validation.py ---------------------------------------
# `collect_reward_matrix` above stores only the per-lambda MEAN over prompts.
# The pre-registered error layer is a PAIRED BOOTSTRAP OVER PROMPTS, which needs
# the per-prompt values. Once Phase B has run with the mean-only collector, those
# values are gone and the CI cannot be recovered without re-running ~9.4 h of
# scoring. Use this collector for Phase B; `collect_reward_matrix` remains for
# callers that genuinely only need means.


def collect_reward_tensor(
    coefficients: np.ndarray,
    rewards_of_lambda: Callable[[np.ndarray], np.ndarray],
    cache_path: str | Path,
    num_prompts: int,
    binding_sha256: str,
) -> np.ndarray:
    """Collect per-prompt rewards with one slice per coefficient vector.

    ``rewards_of_lambda`` must return an array of shape ``(num_prompts,
    num_objectives)`` - the UNAVERAGED scores. The returned tensor has shape
    ``(num_lambda, num_prompts, num_objectives)``; ``tensor.mean(axis=1)``
    reproduces exactly what ``collect_reward_matrix`` would have returned, so
    every downstream consumer of the mean matrix keeps working.

    Caching and resumption follow ``collect_reward_matrix``: one JSONL line per
    completed coefficient, keyed by ``coefficient_key``, flushed immediately so
    an interrupted Colab session resumes instead of restarting.

    ``binding_sha256`` ties the cache to one frozen run. It is written as the
    first line and re-checked on resume. Without it, a cache written under one
    set of adapters, prompts or scorer settings would be silently reused under
    another - the resumption logic would then be actively harmful rather than
    merely useless.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a two-dimensional array.")
    num_rows, num_objectives = coefficients.shape
    if num_prompts < 1:
        raise ValueError("num_prompts must be at least 1.")
    expected_shape = (num_prompts, num_objectives)

    header = {"binding_sha256": str(binding_sha256), "num_prompts": int(num_prompts),
              "num_objectives": int(num_objectives)}
    completed: dict[str, list[list[float]]] = {}
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        lines = [line for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = json.loads(lines[0])
        if "binding_sha256" not in first:
            raise ValueError(
                f"{cache_path} has no binding header. It was written before the run was "
                "bound to a pre-registration. Start a new cache file."
            )
        if first["binding_sha256"] != header["binding_sha256"]:
            raise ValueError(
                "Reward cache belongs to a different run.\n"
                f"  cache:   {first['binding_sha256'][:16]}\n"
                f"  current: {header['binding_sha256'][:16]}\n"
                "Adapters, prompts, matrices, scorer or grids changed since this cache was "
                "written. Use a new RUN_TAG rather than mixing two runs in one file."
            )
        for line in lines[1:]:
            record = json.loads(line)
            completed[str(record["key"])] = record["scores"]
    else:
        cache_path.write_text(json.dumps(header) + "\n", encoding="utf-8")

    tensor = np.full((num_rows, num_prompts, num_objectives), np.nan, dtype=np.float64)
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for index, coefficient in enumerate(coefficients):
            key = coefficient_key(coefficient)
            if key in completed:
                cached = np.asarray(completed[key], dtype=np.float64)
                if cached.shape != expected_shape:
                    raise ValueError(
                        f"Cached entry {key} has shape {cached.shape}, expected "
                        f"{expected_shape}. The prompt set changed since this cache "
                        "was written; start a new cache file rather than mixing them."
                    )
                tensor[index] = cached
                continue

            scores = np.asarray(rewards_of_lambda(coefficient), dtype=np.float64)
            if scores.shape != expected_shape:
                raise ValueError(
                    f"rewards_of_lambda must return shape {expected_shape}, "
                    f"got {scores.shape}."
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError("rewards_of_lambda returned non-finite values.")
            tensor[index] = scores
            cache_file.write(
                json.dumps({
                    "key": key,
                    "index": index,
                    "lambda": coefficient.tolist(),
                    "scores": scores.tolist(),
                    # `reward` is the per-lambda mean under the key that
                    # `collect_reward_matrix` used. Cells 30 and 34 read exactly
                    # this key, so they keep working without modification.
                    "reward": scores.mean(axis=0).tolist(),
                })
                + "\n"
            )
            cache_file.flush()
            print(
                f"[reward] {index + 1}/{num_rows} "
                f"lambda={np.round(coefficient, 3)} "
                f"mean={np.round(scores.mean(axis=0), 4)}"
            )

    if np.isnan(tensor).any():
        raise RuntimeError("Reward tensor still has NaNs after collection.")
    return tensor
