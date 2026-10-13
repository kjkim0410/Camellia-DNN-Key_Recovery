# Camellia_128_master_key_recovery_v3.py 
#
# Full external-key recovery experiment for a ReLU-DNN Camellia-128
# realization, extending the original 64-bit first-round leakage experiment
# # with randomly-generated master key per each iteration.
#
# The original experiment recovers only
#
#     kw1 XOR k1
#
# from the first-round pre-S-box activation.  That 64-bit value is not enough
# to determine the complete 128-bit external key.
#
# This program introduces a synthetic intermediate key-schedule oracle,
# analogous in purpose to the synthetic intermediate-state oracles used for
# AES-192/AES-256 recovery:
#
#     O_KS(X) = X XOR (k3 || k4),       X in R^128.
#
# For Camellia-128, the standard key schedule satisfies
#
#     k3 || k4 = ROTL128(KL, 15),
#     KL       = external 128-bit key.
#
# Symmetric soft queries recover all 128 bits of k3 || k4.  The external key
# is then obtained by the public inverse transformation
#
#     K = ROTR128(k3 || k4, 15).
#
# This is an INTERNAL-ACTIVATION / SYNTHETIC-ORACLE experiment.  It does not
# claim master-key recovery from ordinary plaintext/ciphertext black-box
# access to standard Camellia.
#
# Python 3.8+ compatible.
# Requires Camellia_128_DNN.py in the same directory.

import os
import sys
import random
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Camellia_128_DNN import (
    bytes_to_bits,
    bits_to_int,
    int_to_bits,
    xor_vec,
    generate_subkeys_128_DNN,
    camellia_encrypt_128_DNN,
    camellia_encrypt_128_BOOLEAN,
)


FIXED_KEY_HEX = "0123456789abcdeffedcba9876543210"
TEST_PT_HEX = "0123456789abcdeffedcba9876543210"
EXPECTED_CT_HEX = "67673138549669730857065648eabe43"

MASK128 = (1 << 128) - 1


def rotl128(x, n):
    """Rotate a 128-bit integer left by n positions."""
    n %= 128
    x &= MASK128
    if n == 0:
        return x
    return ((x << n) | (x >> (128 - n))) & MASK128


def rotr128(x, n):
    """Rotate a 128-bit integer right by n positions."""
    n %= 128
    x &= MASK128
    if n == 0:
        return x
    return ((x >> n) | (x << (128 - n))) & MASK128


def join_u64(high, low):
    """Concatenate two unsigned 64-bit words as a 128-bit integer."""
    return ((high & ((1 << 64) - 1)) << 64) | (low & ((1 << 64) - 1))


def symmetric_soft_pair(base_values, target_bit, eps=0.01):
    """
    Return two real-valued vectors differing only at target_bit:

        q_minus[target_bit] = 0.5 - eps
        q_plus[target_bit]  = 0.5 + eps
    """
    if not 0 <= target_bit < len(base_values):
        raise ValueError("target_bit is out of range.")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must satisfy 0 < eps < 0.5.")

    q_minus = [float(x) for x in base_values]
    q_plus = [float(x) for x in base_values]
    q_minus[target_bit] = 0.5 - eps
    q_plus[target_bit] = 0.5 + eps
    return q_minus, q_plus


def get_schedule_words(key_bytes):
    """
    Return integer forms of whitening and round-key words.

    Expected interface of generate_subkeys_128_DNN:
        kw         : four 64-bit bit-vectors [kw1, kw2, kw3, kw4]
        round_keys : eighteen 64-bit bit-vectors [k1, ..., k18]
    """
    kw, round_keys, ke = generate_subkeys_128_DNN(key_bytes)

    if len(kw) < 4:
        raise ValueError("Expected at least four whitening-key words.")
    if len(round_keys) < 4:
        raise ValueError("Expected at least four round-key words.")

    kw_int = [bits_to_int(word) for word in kw]
    rk_int = [bits_to_int(word) for word in round_keys]
    ke_int = [bits_to_int(word) for word in ke]
    return kw_int, rk_int, ke_int


# ---------------------------------------------------------------------------
# Original 64-bit leakage target, retained for comparison
# ---------------------------------------------------------------------------

def first_round_pre_sbox_leakage(pt_bits, key_bytes):
    """Return Z1 = PT_left XOR kw1 XOR k1."""
    if len(pt_bits) != 128:
        raise ValueError("pt_bits must contain exactly 128 values.")

    kw, round_keys, _ = generate_subkeys_128_DNN(key_bytes)
    pt_left = pt_bits[:64]
    l0 = xor_vec(pt_left, kw[0])
    return xor_vec(l0, round_keys[0])


def recover_original_effective_mask(key_bytes, base_pt_bytes, eps=0.01):
    """Recover the original 64-bit target kw1 XOR k1."""
    base_bits = bytes_to_bits(base_pt_bytes)
    recovered = []

    for bit_index in range(64):
        q_minus, q_plus = symmetric_soft_pair(base_bits, bit_index, eps)
        y_minus = first_round_pre_sbox_leakage(q_minus, key_bytes)
        y_plus = first_round_pre_sbox_leakage(q_plus, key_bytes)
        delta = y_plus[bit_index] - y_minus[bit_index]

        if delta > 0.0:
            recovered.append(0.0)
        elif delta < 0.0:
            recovered.append(1.0)
        else:
            raise RuntimeError("Zero delta in original leakage attack.")

    return bits_to_int(recovered)


# ---------------------------------------------------------------------------
# New synthetic intermediate key-schedule oracle
# ---------------------------------------------------------------------------

def key_schedule_pair_k3_k4_bits(key_bytes):
    """
    Return the 128-bit vector k3 || k4.

    Camellia-128 schedule identity:

        k3 || k4 = ROTL128(external_key, 15).
    """
    _, round_keys, _ = generate_subkeys_128_DNN(key_bytes)
    if len(round_keys) < 4:
        raise ValueError("round_keys must contain k1 through k4.")
    return list(round_keys[2]) + list(round_keys[3])


def synthetic_key_schedule_oracle(probe_bits, key_bytes):
    """
    Synthetic intermediate oracle

        O_KS(X) = X XOR (k3 || k4).

    probe_bits may contain binary or real values.  The XOR implementation is
    the same ReLU-compatible xor_vec used by the DNN Camellia realization.
    """
    if len(probe_bits) != 128:
        raise ValueError("probe_bits must contain exactly 128 values.")

    target = key_schedule_pair_k3_k4_bits(key_bytes)
    return xor_vec(probe_bits, target)


def recover_k3_k4_once(key_bytes, base_probe_bits=None, eps=0.01, verbose=True):
    """
    Recover all 128 bits of k3 || k4 using 128 symmetric soft-query pairs.

    For y = XOR(x, s):
        s=0 -> y increases with x -> positive delta
        s=1 -> y decreases with x -> negative delta
    """
    if base_probe_bits is None:
        base_probe_bits = [0.0] * 128
    if len(base_probe_bits) != 128:
        raise ValueError("base_probe_bits must contain exactly 128 values.")

    recovered_bits = []
    deltas = []

    for bit_index in range(128):
        q_minus, q_plus = symmetric_soft_pair(
            base_probe_bits, bit_index, eps
        )
        y_minus = synthetic_key_schedule_oracle(q_minus, key_bytes)
        y_plus = synthetic_key_schedule_oracle(q_plus, key_bytes)
        delta = y_plus[bit_index] - y_minus[bit_index]
        deltas.append(delta)

        if delta > 0.0:
            guessed = 0
        elif delta < 0.0:
            guessed = 1
        else:
            raise RuntimeError(
                "Zero oracle delta at bit %d; increase eps." % bit_index
            )

        recovered_bits.append(float(guessed))

        if verbose:
            print(
                "bit %3d | oracle-=%.6f oracle+=%.6f "
                "delta=%+.6f | recovered=%d"
                % (
                    bit_index,
                    y_minus[bit_index],
                    y_plus[bit_index],
                    delta,
                    guessed,
                )
            )

    return bits_to_int(recovered_bits), deltas


def recover_external_key_from_k3_k4(recovered_k3_k4):
    """Invert k3||k4 = ROTL128(K,15)."""
    return rotr128(recovered_k3_k4, 15)


def recover_external_key_once(
    key_bytes,
    base_probe_bits=None,
    eps=0.01,
    verbose=True,
):
    """Recover k3||k4 and then reconstruct the external 128-bit key."""
    recovered_pair, deltas = recover_k3_k4_once(
        key_bytes=key_bytes,
        base_probe_bits=base_probe_bits,
        eps=eps,
        verbose=verbose,
    )
    recovered_key_int = recover_external_key_from_k3_k4(recovered_pair)
    return recovered_key_int, recovered_pair, deltas


# ---------------------------------------------------------------------------
# Independent validation helpers
# ---------------------------------------------------------------------------

def true_schedule_values(key_bytes):
    """Ground truth used only to validate the experiment."""
    kw, rk, _ = get_schedule_words(key_bytes)
    kw1_kw2 = join_u64(kw[0], kw[1])
    k3_k4 = join_u64(rk[2], rk[3])
    return kw1_kw2, k3_k4


def verify_schedule_identity(key_bytes):
    """Check k3||k4 = ROTL128(K,15) for the supplied implementation."""
    external_key = int.from_bytes(key_bytes, byteorder="big")
    kw1_kw2, k3_k4 = true_schedule_values(key_bytes)
    expected_pair = rotl128(external_key, 15)

    print("=== Camellia-128 key-schedule verification ===")
    print("External key          = %032x" % external_key)
    print("kw1 || kw2            = %032x" % kw1_kw2)
    print("k3  || k4             = %032x" % k3_k4)
    print("ROTL128(key, 15)      = %032x" % expected_pair)
    print("kw1||kw2 == key       =", kw1_kw2 == external_key)
    print("k3||k4 identity       =", k3_k4 == expected_pair)

    if kw1_kw2 != external_key:
        raise AssertionError(
            "Implementation does not map kw1||kw2 directly to the key."
        )
    if k3_k4 != expected_pair:
        raise AssertionError(
            "Implementation round-key ordering differs from the expected "
            "Camellia-128 k3||k4 schedule identity."
        )


def verify_cipher_implementation(key_bytes):
    """Check Boolean and DNN implementations against the Camellia KAT."""
    pt = bytes.fromhex(TEST_PT_HEX)
    expected = bytes.fromhex(EXPECTED_CT_HEX)

    ct_boolean = camellia_encrypt_128_BOOLEAN(pt, key_bytes)
    ct_dnn = camellia_encrypt_128_DNN(pt, key_bytes)

    print("=== Cipher verification ===")
    print("PT                    =", pt.hex())
    print("KEY                   =", key_bytes.hex())
    print("CT Boolean            =", ct_boolean.hex())
    print("CT DNN                =", ct_dnn.hex())
    print("CT expected           =", expected.hex())
    print("Boolean == DNN        =", ct_boolean == ct_dnn)
    print("DNN matches KAT       =", ct_dnn == expected)

    if ct_boolean != ct_dnn or ct_dnn != expected:
        raise AssertionError("Camellia implementation verification failed.")

def random_key_trials(
    num_trials=100,
    eps=0.01,
):
    """
    Each trial:
      1. generate a fresh random Camellia-128 key
      2. generate a fresh random probe
      3. recover the external 128-bit key
      4. compare recovered key with the randomly generated key
    """
    successes = 0
    start = time.perf_counter()

    for trial in range(1, num_trials + 1):

        # Fresh random 128-bit Camellia key for EVERY trial
        key_bytes = os.urandom(16)

        # Fresh random 128-bit binary probe
        probe_bytes = os.urandom(16)
        probe_int = int.from_bytes(probe_bytes, byteorder="big")

        probe = [
            float((probe_int >> (127 - i)) & 1)
            for i in range(128)
        ]

        expected_key = int.from_bytes(key_bytes, byteorder="big")

        recovered_key, recovered_pair, _ = recover_external_key_once(
            key_bytes=key_bytes,
            base_probe_bits=probe,
            eps=eps,
            verbose=False,
        )

        ok = recovered_key == expected_key
        successes += int(ok)

        print(
            "trial %3d/%3d | target key=%032x | "
            "recovered key=%032x | match=%s"
            % (
                trial,
                num_trials,
                expected_key,
                recovered_key,
                ok,
            )
        )

    return successes, time.perf_counter() - start


def main():
    key = bytes.fromhex(FIXED_KEY_HEX)
    base_pt = bytes.fromhex(TEST_PT_HEX)
    eps = 0.01

    verify_cipher_implementation(key)
    print()
    verify_schedule_identity(key)

    kw, rk, _ = get_schedule_words(key)
    true_original_mask = kw[0] ^ rk[0]
    recovered_original_mask = recover_original_effective_mask(
        key_bytes=key,
        base_pt_bytes=base_pt,
        eps=eps,
    )

    print("\n=== Original 64-bit attack retained for comparison ===")
    print("True kw1 XOR k1       = %016x" % true_original_mask)
    print("Recovered mask        = %016x" % recovered_original_mask)
    print("64-bit mask recovery  =", recovered_original_mask == true_original_mask)

    print("\n=== New full master-key recovery ===")
    print("Oracle                = X XOR (k3 || k4)")
    print("Schedule relation     = k3 || k4 = ROTL128(K, 15)")
    print("Soft-query epsilon    =", eps)
    print("Query pairs           = 128")
    print()

    recovered_key, recovered_pair, deltas = recover_external_key_once(
        key_bytes=key,
        base_probe_bits=[0.0] * 128,
        eps=eps,
        verbose=True,
    )

    expected_key = int.from_bytes(key, byteorder="big")
    _, true_pair = true_schedule_values(key)

    print("\n=== Single-trial result ===")
    print("Recovered k3 || k4    = %032x" % recovered_pair)
    print("Expected  k3 || k4    = %032x" % true_pair)
    print("Recovered external K  = %032x" % recovered_key)
    print("Expected  external K  = %032x" % expected_key)
    print("Round-key pair match  =", recovered_pair == true_pair)
    print("Full 128-bit recovery =", recovered_key == expected_key)
    print("Minimum |delta|       = %.6f" % min(abs(x) for x in deltas))
    print("Maximum |delta|       = %.6f" % max(abs(x) for x in deltas))

    if recovered_pair != true_pair or recovered_key != expected_key:
        raise AssertionError("Camellia-128 external-key recovery failed.")

    print("\n=== Random-key repeated validation ===")

###################################
# Decide the number of trials here 
##################################
    num_trials = 1000

    successes, elapsed = random_key_trials(
        num_trials=num_trials,
        eps=eps,
    )

    print("\n=== Final summary ===")
    print("Target keys           = fresh random Camellia-128 key per trial")
    print("Recovered target      = complete Camellia-128 external key")
    print("Recovery method       = recover k3||k4, then ROTR128(...,15)")
    print("Trials                =", num_trials)
    print("Successes             = %d/%d" % (successes, num_trials))
    print("Success rate          = %.2f%%" %
        (100.0 * successes / num_trials))
    print("Elapsed seconds       = %.6f" % elapsed)
    print("MASTER-KEY TEST       =",
          "PASS" if successes == num_trials else "FAIL")


    if successes != num_trials:
        raise AssertionError("Repeated master-key recovery validation failed.")


if __name__ == "__main__":
    main()
