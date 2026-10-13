# Camellia_256_master_key_recovery_v1.py
#
# Camellia-256 external master-key recovery experiment based on the
# key-schedule identities
#
#   k3 || k4   = ROTL128(KR, 15)
#   k9 || k10  = ROTL128(KL, 45)
#
# where the 256-bit external key is K = KL || KR.
#
# Thus:
#   KR = ROTR128(k3 || k4, 15)
#   KL = ROTR128(k9 || k10, 45)
#
# This program uses the same synthetic/internal key-schedule oracle idea
# as the Camellia-128 master-key-recovery experiment. It is NOT a
# plaintext/ciphertext-only black-box key-recovery attack.
#
# Python 3.8+; no external packages required.

import argparse
import importlib.util
import secrets
from pathlib import Path

MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1

DEFAULT_PT_HEX = "0123456789abcdeffedcba9876543210"
EXPECTED_CT_HEX = "9acc237dff16d76c20ef7c919e3a7509"


def load_camellia256_module(path=None):
    """Load the companion Camellia_256_DNN implementation."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))

    here = Path(__file__).resolve().parent
    candidates += [
        here / "Camellia_256_DNN.py",
        here / "Camellia_256_DNN(1).py",
    ]

    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "camellia_256_dnn_module", str(candidate)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, candidate

    raise FileNotFoundError(
        "Cannot find Camellia_256_DNN.py or Camellia_256_DNN(1).py. "
        "Place it in the same directory, or pass --dnn-file PATH."
    )


def rotl128(x, n):
    n %= 128
    return ((x << n) & MASK128) | ((x & MASK128) >> ((128 - n) % 128))


def rotr128(x, n):
    n %= 128
    return ((x & MASK128) >> n) | ((x << ((128 - n) % 128)) & MASK128)


def join_u64(hi, lo):
    return ((hi & MASK64) << 64) | (lo & MASK64)


def int_to_soft_bits(x, nbits=128):
    return [float((x >> (nbits - 1 - i)) & 1) for i in range(nbits)]


def bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit >= 0.5 else 0)
    return value


def recover_external_key_from_subkeys(k3k4, k9k10):
    """Recover KL, KR and K = KL || KR from the two 128-bit subkey pairs."""
    KR = rotr128(k3k4, 15)
    KL = rotr128(k9k10, 45)
    K = (KL << 128) | KR
    return KL, KR, K


class KeyScheduleXorOracle:
    """
    Synthetic internal key-schedule oracle.

    For target T in {k3||k4, k9||k10}, return X XOR T over soft
    real-valued bits using the exact ReLU XOR realization:

        XOR(x,b) = ReLU(x-b) + ReLU(b-x), b in {0,1}.

    Around x=0.5:
        if b=0, O(0.5+eps)-O(0.5-eps) = +2 eps
        if b=1, O(0.5+eps)-O(0.5-eps) = -2 eps

    Therefore the sign of the symmetric response difference reveals b.
    """

    def __init__(self, target_bits):
        if len(target_bits) != 128:
            raise ValueError("Oracle target must contain 128 bits.")
        self.target_bits = [float(b) for b in target_bits]

    @staticmethod
    def relu(x):
        return x if x > 0.0 else 0.0

    @classmethod
    def relu_xor(cls, x, b):
        return cls.relu(x - b) + cls.relu(b - x)

    def query_bit(self, bit_index, x):
        return self.relu_xor(float(x), self.target_bits[bit_index])


def recover_128bit_target(oracle, eps=0.10, verbose=False, label="target"):
    """
    Recover one 128-bit target using 128 symmetric query pairs.

    delta = O_i(0.5+eps) - O_i(0.5-eps)
      delta > 0 -> target bit 0
      delta < 0 -> target bit 1
    """
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must satisfy 0 < eps < 0.5.")

    recovered = []
    for i in range(128):
        minus = oracle.query_bit(i, 0.5 - eps)
        plus = oracle.query_bit(i, 0.5 + eps)
        delta = plus - minus
        bit = 0 if delta > 0.0 else 1
        recovered.append(float(bit))

        if verbose:
            print(
                "%s bit %3d: O-=%.6f  O+=%.6f  delta=%+.6f  recovered=%d"
                % (label, i, minus, plus, delta, bit)
            )

    return bits_to_int(recovered)


def true_schedule_values(cam, key_bytes):
    """
    Obtain the two target subkey pairs from the companion Camellia-256
    DNN schedule.
    """
    _, k, _ = cam.generate_subkeys_256_DNN(key_bytes)

    k3 = cam.bits_to_int(k[2])
    k4 = cam.bits_to_int(k[3])
    k9 = cam.bits_to_int(k[8])
    k10 = cam.bits_to_int(k[9])

    k3k4 = join_u64(k3, k4)
    k9k10 = join_u64(k9, k10)
    return k3, k4, k9, k10, k3k4, k9k10


def verify_schedule_identity(cam, key_bytes, verbose=True):
    """
    Verify:
        k3||k4  = ROTL128(KR,15)
        k9||k10 = ROTL128(KL,45)
    """
    KL = int.from_bytes(key_bytes[:16], "big")
    KR = int.from_bytes(key_bytes[16:], "big")

    k3, k4, k9, k10, k3k4, k9k10 = true_schedule_values(cam, key_bytes)

    ok_r = (k3k4 == rotl128(KR, 15))
    ok_l = (k9k10 == rotl128(KL, 45))

    if verbose:
        print("=== Camellia-256 key-schedule identity check ===")
        print("KL                 = %032x" % KL)
        print("KR                 = %032x" % KR)
        print("k3 || k4           = %032x" % k3k4)
        print("ROTL128(KR,15)     = %032x" % rotl128(KR, 15))
        print("k9 || k10          = %032x" % k9k10)
        print("ROTL128(KL,45)     = %032x" % rotl128(KL, 45))
        print("KR identity        =", ok_r)
        print("KL identity        =", ok_l)
        print()

    if not (ok_r and ok_l):
        raise AssertionError("Camellia-256 schedule identity check failed.")

    return True


def verify_cipher_implementation(cam, key_bytes, pt_bytes=None, verbose=True):
    """
    Check the companion Boolean and ReLU-DNN implementations against each
    other. For the official fixed test key, also check the known vector.
    """
    if pt_bytes is None:
        pt_bytes = bytes.fromhex(DEFAULT_PT_HEX)

    ct_boolean = cam.camellia_encrypt_256_BOOLEAN(pt_bytes, key_bytes)
    ct_dnn = cam.camellia_encrypt_256_DNN(pt_bytes, key_bytes)

    if verbose:
        print("=== Boolean vs ReLU-DNN encryption check ===")
        print("PT           =", pt_bytes.hex())
        print("KEY          =", key_bytes.hex())
        print("CT Boolean   =", ct_boolean.hex())
        print("CT DNN       =", ct_dnn.hex())
        print("Boolean==DNN =", ct_boolean == ct_dnn)

    if ct_boolean != ct_dnn:
        raise AssertionError("Boolean and DNN Camellia-256 implementations disagree.")

    official_key = bytes.fromhex(
        "0123456789abcdeffedcba9876543210"
        "00112233445566778899aabbccddeeff"
    )
    official_pt = bytes.fromhex(DEFAULT_PT_HEX)
    if key_bytes == official_key and pt_bytes == official_pt:
        expected = bytes.fromhex(EXPECTED_CT_HEX)
        if verbose:
            print("CT expected  =", expected.hex())
            print("Official OK  =", ct_boolean == expected)
        if ct_boolean != expected:
            raise AssertionError("Official Camellia-256 test vector failed.")

    if verbose:
        print()
    return True


def recover_external_key_once(cam, key_bytes, eps=0.10,
                              verbose_bits=False, verbose=True):
    """
    Recover the complete 256-bit external key.

    Stage 1: recover k3||k4, then KR = ROTR128(...,15)
    Stage 2: recover k9||k10, then KL = ROTR128(...,45)
    Stage 3: concatenate KL||KR.
    """
    if len(key_bytes) != 32:
        raise ValueError("Camellia-256 requires a 32-byte external key.")

    KL_true = int.from_bytes(key_bytes[:16], "big")
    KR_true = int.from_bytes(key_bytes[16:], "big")
    K_true = int.from_bytes(key_bytes, "big")

    _, _, _, _, k3k4_true, k9k10_true = true_schedule_values(cam, key_bytes)

    oracle_r = KeyScheduleXorOracle(int_to_soft_bits(k3k4_true, 128))
    oracle_l = KeyScheduleXorOracle(int_to_soft_bits(k9k10_true, 128))

    k3k4_hat = recover_128bit_target(
        oracle_r, eps=eps, verbose=verbose_bits, label="k3||k4"
    )
    k9k10_hat = recover_128bit_target(
        oracle_l, eps=eps, verbose=verbose_bits, label="k9||k10"
    )

    KL_hat, KR_hat, K_hat = recover_external_key_from_subkeys(
        k3k4_hat, k9k10_hat
    )

    ok_k3k4 = (k3k4_hat == k3k4_true)
    ok_k9k10 = (k9k10_hat == k9k10_true)
    ok_kl = (KL_hat == KL_true)
    ok_kr = (KR_hat == KR_true)
    ok_key = (K_hat == K_true)

    if verbose:
        print("=== Camellia-256 external-key recovery ===")
        print("True k3||k4        = %032x" % k3k4_true)
        print("Recovered k3||k4   = %032x" % k3k4_hat)
        print("True k9||k10       = %032x" % k9k10_true)
        print("Recovered k9||k10  = %032x" % k9k10_hat)
        print()
        print("True KR             = %032x" % KR_true)
        print("Recovered KR        = %032x" % KR_hat)
        print("True KL             = %032x" % KL_true)
        print("Recovered KL        = %032x" % KL_hat)
        print()
        print("True external K     = %064x" % K_true)
        print("Recovered external K= %064x" % K_hat)
        print()
        print("k3||k4 recovered    =", ok_k3k4)
        print("k9||k10 recovered   =", ok_k9k10)
        print("KR recovered        =", ok_kr)
        print("KL recovered        =", ok_kl)
        print("MASTER KEY RECOVERY =", "SUCCESS" if ok_key else "FAIL")
        print()

    return {
        "true_key": K_true,
        "recovered_key": K_hat,
        "true_KL": KL_true,
        "recovered_KL": KL_hat,
        "true_KR": KR_true,
        "recovered_KR": KR_hat,
        "true_k3k4": k3k4_true,
        "recovered_k3k4": k3k4_hat,
        "true_k9k10": k9k10_true,
        "recovered_k9k10": k9k10_hat,
        "success": ok_key,
    }


def run_trials(cam, trials=20, eps=0.10, show_each=True):
    successes = 0

    print("=== Repeated random-key experiment ===")
    print("Trials =", trials)
    print("eps    =", eps)
    print()

    for t in range(1, trials + 1):
        key = secrets.token_bytes(32)
        result = recover_external_key_once(
            cam, key, eps=eps, verbose_bits=False, verbose=False
        )
        successes += int(result["success"])

        if show_each:
            print(
                "Trial %3d/%d : %s  K=%s"
                % (t, trials,
                   "SUCCESS" if result["success"] else "FAIL",
                   key.hex())
            )

    print()
    print("Successful trials = %d/%d" % (successes, trials))
    print("Success rate      = %.2f%%" % (100.0 * successes / trials))
    return successes == trials


def main():
    parser = argparse.ArgumentParser(
        description="Camellia-256 synthetic key-schedule master-key recovery."
    )
    parser.add_argument(
        "--dnn-file",
        default=None,
        help="Path to Camellia_256_DNN.py (default: same directory).",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="64-hex-digit Camellia-256 key. Default: fresh random key.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.10,
        help="Symmetric soft-query perturbation epsilon (default: 0.10).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        ## Set the number of test trial, default as you want ##
        default=1000,
        help="Number of fresh random-key recovery trials (default: 1000).",
        # default=20,
        # help="Number of fresh random-key recovery trials (default: 20).",
    )
    parser.add_argument(
        "--verbose-bits",
        action="store_true",
        help="Print all 256 recovered target bits.",
    )
    parser.add_argument(
        "--skip-cipher-check",
        action="store_true",
        help="Skip Boolean-vs-DNN encryption verification.",
    )
    args = parser.parse_args()

    cam, loaded_path = load_camellia256_module(args.dnn_file)
    print("Loaded DNN module:", loaded_path)
    print()

    if args.key is None:
        key = secrets.token_bytes(32)
    else:
        try:
            key = bytes.fromhex(args.key)
        except ValueError:
            raise SystemExit("--key must be a hexadecimal string.")
        if len(key) != 32:
            raise SystemExit("--key must contain exactly 64 hex digits (32 bytes).")

    print("Selected key =", key.hex())
    print()

    verify_schedule_identity(cam, key, verbose=True)

    if not args.skip_cipher_check:
        verify_cipher_implementation(cam, key, verbose=True)

    result = recover_external_key_once(
        cam,
        key,
        eps=args.eps,
        verbose_bits=args.verbose_bits,
        verbose=True,
    )
    if not result["success"]:
        raise SystemExit("Single-key master-key recovery failed.")

    if args.trials > 0:
        all_ok = run_trials(cam, trials=args.trials, eps=args.eps)
        if not all_ok:
            raise SystemExit("At least one repeated recovery trial failed.")


if __name__ == "__main__":
    main()
