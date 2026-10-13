# Reproducibility guide

1. Keep the included original 128-bit and 256-bit companion modules beside the scripts. Do not substitute the 192-bit implementation without verifying its interfaces and schedule.
2. Record the repository commit, Python version, OS, CPU, and parameter changes.
3. Run the embedded correctness checks and then 1,000 random-key trials for each variant using the README commands.
4. Save complete output and inspect the final recovery summary. For Camellia-192, also inspect complement-consistency failures. Do not infer success solely from process completion: its repeated-trial function returns success/failure counts and its entry point does not assert that failures are zero.
5. Report the exact recovered-key count and denominator. The initial demonstrations and known-answer tests are separate from the repeated random-key trials.

The scripts sample fresh random keys using OS-backed randomness (`os.urandom` or `secrets.token_bytes`). Exact trial sequences are not controlled by a fixed seed. New runs test the same protocol with different keys. Keep logs to document the particular run.

The published result to reproduce is 1,000/1,000 exact recoveries for each variant, with zero Camellia-192 complement-consistency failures. This result was provided by the authors; the original full-run logs are not supplied here. Do not claim execution times or hardware-specific performance without measurements.

Camellia-128/256 test reconstruction through synthetic internal oracles initialized with true target subkey bits. Camellia-192 tests reconstruction from directly accessible DNN round keys. Functional agreement on binary inputs and key reconstruction under these access models are distinct claims. These experiments do not evaluate noisy leakage, ciphertext-only key recovery, or obtaining the assumed oracle from a protected implementation.
