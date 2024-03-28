from hashlib import sha256
from pathlib import Path

def find_model_in_cache(model_cache: Path, model_name: str, model_hash: str) -> Path:
    best_model_filename = None
    for hash_char_count in range(0, len(model_hash)):
        if hash_char_count == 0:
            model_filename = model_cache / f"{model_name}.stan"
        else:
            model_filename = model_cache / f"{model_name} {model_hash[:hash_char_count]}.stan"
        if model_filename.exists():
            # load the model and check its hash if it matches
            with model_filename.open("r") as f:
                model_code = f.read()
            if sha256(model_code.encode()).hexdigest() == model_hash:
                if hash_char_count == len(model_hash) - 1:
                    raise RuntimeError(
                        f"Hash collision for model {model_name} with hash {model_hash} in cache file {model_filename}!")
                return model_filename
        else:
            if best_model_filename is None:
                best_model_filename = model_filename

        hash_char_count += 1

    assert best_model_filename is not None
    return best_model_filename
