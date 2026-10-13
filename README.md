# Camellia ReLU Master-Key Reconstruction Experiments

Research code for functional correctness and conditional master-key reconstruction in explicitly constructed ReLU implementations of Camellia-128, Camellia-192, and Camellia-256.

**Scope:** Camellia-128 and Camellia-256 use explicitly constructed synthetic internal key-schedule oracles. Camellia-192 directly accesses DNN-generated round keys. These experiments do not demonstrate plaintext/ciphertext-only attacks against standard Camellia.

## Reported results

The authors report the following random-key experiment results:

| Variant | Random-key trials | Exact recoveries | Failures | Success rate | Access model |
|---|---:|---:|---:|---:|---|
| Camellia-128 | 1,000 | 1,000 | 0 | 100% | Synthetic internal oracle |
| Camellia-192 | 1,000 | 1,000 | 0 | 100% | Direct DNN round-key access |
| Camellia-256 | 1,000 | 1,000 | 0 | 100% | Synthetic internal oracles |
| **Total** | **3,000** | **3,000** | **0** | **100%** | |

Success means exact equality of the reconstructed external key and the randomly sampled original key. Camellia-192 additionally checks the complement relation in its internal `KR` value; no complement-consistency failures were reported. Exact recovery implies zero master-key Hamming distance.

These are author-reported results. Original 1,000-trial execution logs are not included in this package. The observed rates concern the implemented access models and do not measure robustness to noisy physical leakage or demonstrate extraction of subkeys from a deployed implementation.

## Files and required companion modules

| File | Purpose | Required companion |
|---|---|---|
| `Camellia_128_master_key_recovery_v4_1000times.py` | Recover one 128-bit subkey target, then invert a rotation | `Camellia_128_DNN.py` |
| `Camellia_192_master_key_recovery_v2_1000times.py` | Generate DNN round keys and invert the schedule | None; implementation is embedded |
| `Camellia_256_master_key_recovery_v1_1000times.py` | Recover two 128-bit targets, then reconstruct both key components | `Camellia_256_DNN.py` |
| `docs/REPRODUCIBILITY.md` | Experiment protocol and interpretation | — |
| `docs/PUBLISHING_CHECKLIST.md` | Release preparation and license guidance | — |
| `requirements.txt` | Dependency note | — |

**Both required companion modules are included in this package.** Keep them beside the experiment scripts. The 256-bit loader also accepts `Camellia_256_DNN(1).py` or a path supplied through `--dnn-file`; the 128-bit script requires the exact importable name `Camellia_128_DNN.py`.

The experiment scripts are copies of the supplied files with the upload suffix `(1)` removed from their filenames. Their source code is unchanged. The companion modules are also unchanged copies, renamed to their importable filenames. Some original header comments mention earlier versions; the filenames identify the supplied experiment versions.

## Package validation

During package preparation, all three variants passed their known-answer checks and three fresh random-key recovery trials each. These short checks verify execution compatibility; the 1,000-trial results above remain the authors' reported experiments.

## Requirements

The scripts declare Python 3.8+ compatibility. Python 3.10 is a suitable environment for reproducing the authors' workflow. The three supplied scripts use only the Python standard library. The included companion modules also require no external packages.

No training framework, GPU, NumPy, or PyTorch is required by these three scripts themselves. Their ReLU gates and selectors are explicitly constructed computations rather than learned networks.

## Run the experiments

Run commands from the repository directory. On Windows, `py -3.10` may be used instead of `python` to select the intended interpreter.

### Camellia-192 (self-contained)

```bash
python Camellia_192_master_key_recovery_v2_1000times.py
```

The entry point first runs the embedded known-answer test, then performs 1,000 random-key recovery trials. Change `NUM_TRIALS` in the entry point for a shorter run.

### Camellia-128 (companion module required)

```bash
python Camellia_128_master_key_recovery_v4_1000times.py
```

The entry point includes encryption and schedule checks, a demonstration, and 1,000 random-key trials. Change `num_trials = 1000` in `main()` for a shorter run. This script has no command-line trial option. Its perturbation is `eps = 0.01`.

### Camellia-256 (companion module required)

```bash
python Camellia_256_master_key_recovery_v1_1000times.py --trials 1000
```

A shorter initial check:

```bash
python Camellia_256_master_key_recovery_v1_1000times.py --trials 10
```

An explicit companion-module path:

```bash
python Camellia_256_master_key_recovery_v1_1000times.py --dnn-file ./Camellia_256_DNN.py --trials 1000 --eps 0.10
```

Run `--help` for all options. The script performs an initial demonstration in addition to the requested random-key trials. `--trials 0` skips the repeated trial loop but still executes that demonstration. Keep the default cipher check enabled for validation; `--skip-cipher-check` omits it.

## Reconstruction method

For a target bit `b` and a real-valued probe `x`, the synthetic oracle evaluates

```text
O_b(x) = ReLU(x - b) + ReLU(b - x) = |x - b|.
Delta = O_b(1/2 + eps) - O_b(1/2 - eps) = 2 eps (1 - 2b).
```

The sign of `Delta` identifies the bit. The target bits are supplied from the internal schedule when constructing the oracle; this is an access assumption, rather than a procedure for obtaining that access from ciphertexts.

| Variant | Target material | Reconstruction | Symmetric query pairs |
|---|---|---|---:|
| 128 | `k3 || k4` | `K = ROTR128(k3 || k4, 15)` | 128 |
| 192 | Directly available `k3 || k4` and `k9 || k10` | Recover `KR` and `KL`; concatenate `KL` with the upper 64 bits of `KR` | Not applicable |
| 256 | `k3 || k4` and `k9 || k10` | `K = ROTR128(k9 || k10, 45) || ROTR128(k3 || k4, 15)` | 256 |

Here `||` means concatenation and round-key numbering is one-based. For Camellia-192, `KR = R || complement(R)`. Recovering only the 128-bit pre-whitening value `KL` leaves 64 unknown bits for Camellia-192 and 128 for Camellia-256.

The query counts above describe the coordinate-wise procedures as implemented, not a lower bound for all possible oracle-query strategies. The 128-bit procedure uses 256 vector oracle calls and selects the probed output coordinate; the 256-bit procedure uses 512 scalar bit-oracle calls. The Camellia-192 repeated experiment does not use perturbation-based subkey extraction.

## Capture execution logs

Create a local `results` directory and retain complete stdout/stderr, Python version, platform details, and any modified parameters. For example, in PowerShell:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
python --version
python Camellia_192_master_key_recovery_v2_1000times.py 2>&1 | Tee-Object -FilePath results/camellia192.txt
```

See [the reproducibility guide](docs/REPRODUCIBILITY.md) for interpretation and additional checks.

## Troubleshooting

- `ModuleNotFoundError: Camellia_128_DNN`: add the original companion module with that exact filename beside the 128-bit script.
- `Cannot find Camellia_256_DNN.py`: add the original companion module or pass its path with `--dnn-file`.
- A different Python installation runs on Windows: use `py -3.10` and inspect `python -c "import sys; print(sys.executable)"`.
- Large output: the scripts print generated experiment keys and progress; redirect output to a file. Use only generated research keys in these demonstrations.

## Publication and reuse

Suggested repository name: `camellia-relu-key-reconstruction`. Suggested description: "Exact ReLU Camellia implementations and internal-access master-key reconstruction experiments for 128-, 192-, and 256-bit keys."

This repository is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Kwangjo Kim.

Paper title, complete author list, and publication URL/DOI should be finalized before release. See [the publishing checklist](docs/PUBLISHING_CHECKLIST.md).
