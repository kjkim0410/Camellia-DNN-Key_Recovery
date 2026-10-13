# Camellia_192_DNN.py
# ReLU-DNN realization of Camellia-192 for binary inputs.
#
# The Boolean gates are represented exactly by ReLU units:
#   AND(x,y) = ReLU(x+y-1)
#   XOR(x,y) = ReLU(x-y) + ReLU(y-x)
#   OR(x,y)  = x+y-AND(x,y)
#   NOT(x)   = 1-x
#
# The Camellia S-box is represented as a 256-way ReLU selector network.
# For an 8-bit binary input x and candidate byte a:
#   match_a(x) = ReLU(sum_i [x_i if a_i=1 else (1-x_i)] - 7)
# This is exactly 1 for x=a and 0 for every other binary byte.
#
# Python 3.8+ compatible. No external packages are required.

MASK8 = 0xFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
MASK128 = (1 << 128) - 1

SBOX1 = [
    112,130,44,236,179,39,192,229,228,133,87,53,234,12,174,65,
    35,239,107,147,69,25,165,33,237,14,79,78,29,101,146,189,
    134,184,175,143,124,235,31,206,62,48,220,95,94,197,11,26,
    166,225,57,202,213,71,93,61,217,1,90,214,81,86,108,77,
    139,13,154,102,251,204,176,45,116,18,43,32,240,177,132,153,
    223,76,203,194,52,126,118,5,109,183,169,49,209,23,4,215,
    20,88,58,97,222,27,17,28,50,15,156,22,83,24,242,34,
    254,68,207,178,195,181,122,145,36,8,232,168,96,252,105,80,
    170,208,160,125,161,137,98,151,84,91,30,149,224,255,100,210,
    16,196,0,72,163,247,117,219,138,3,230,218,9,63,221,148,
    135,92,131,2,205,74,144,51,115,103,246,243,157,127,191,226,
    82,155,216,38,200,55,198,59,129,150,111,75,19,190,99,46,
    233,121,167,140,159,110,188,142,41,245,249,182,47,253,180,89,
    120,152,6,106,231,70,113,186,212,37,171,66,136,162,141,250,
    114,7,185,85,248,238,172,10,54,73,42,104,60,56,241,164,
    64,40,211,123,187,201,67,193,21,227,173,244,119,199,128,158
]

SIGMA = [
    0xA09E667F3BCC908B,
    0xB67AE8584CAA73B2,
    0xC6EF372FE94F82BE,
    0x54FF53A5F1D36F1C,
    0x10E527FADE682D1D,
    0xB05688C2B3E6C1FD,
]


# ----------------------------------------------------------------------
# ReLU neurons and exact Boolean gates on {0,1}
# ----------------------------------------------------------------------

def relu(x):
    return x if x > 0.0 else 0.0

def dnn_not(x):
    return 1.0 - x

def dnn_and(x, y):
    return relu(x + y - 1.0)

def dnn_or(x, y):
    return x + y - dnn_and(x, y)

def dnn_xor(x, y):
    return relu(x - y) + relu(y - x)

def xor_vec(a, b):
    return [dnn_xor(x, y) for x, y in zip(a, b)]

def and_vec(a, b):
    return [dnn_and(x, y) for x, y in zip(a, b)]

def or_vec(a, b):
    return [dnn_or(x, y) for x, y in zip(a, b)]

def xor_many(*vectors):
    out = list(vectors[0])
    for v in vectors[1:]:
        out = xor_vec(out, v)
    return out


# ----------------------------------------------------------------------
# Bit conversion and wire permutations
# ----------------------------------------------------------------------

def int_to_bits(x, nbits):
    return [float((x >> (nbits - 1 - i)) & 1) for i in range(nbits)]

def bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit >= 0.5 else 0)
    return value

def bytes_to_bits(data):
    bits = []
    for b in data:
        bits.extend(int_to_bits(b, 8))
    return bits

def bits_to_bytes(bits):
    if len(bits) % 8 != 0:
        raise ValueError("Bit length must be a multiple of 8.")
    return bytes(bits_to_int(bits[i:i+8]) for i in range(0, len(bits), 8))

def rotl_bits(bits, n):
    n %= len(bits)
    return bits[n:] + bits[:n]

def split128_bits(bits):
    return bits[:64], bits[64:]


# ----------------------------------------------------------------------
# ReLU S-box selector network
# ----------------------------------------------------------------------

BYTE_PATTERNS = [int_to_bits(a, 8) for a in range(256)]

def rol8_int(x, n):
    n %= 8
    return ((x << n) & MASK8) | ((x & MASK8) >> (8 - n))

def s1_int(x):
    return SBOX1[x]

def s2_int(x):
    return rol8_int(SBOX1[x], 1)

def s3_int(x):
    return rol8_int(SBOX1[x], 7)

def s4_int(x):
    return SBOX1[rol8_int(x, 1)]

def make_sbox_bit_table(sbox_function):
    return [
        [float((sbox_function(a) >> (7 - output_bit)) & 1) for a in range(256)]
        for output_bit in range(8)
    ]

S1_TABLE = make_sbox_bit_table(s1_int)
S2_TABLE = make_sbox_bit_table(s2_int)
S3_TABLE = make_sbox_bit_table(s3_int)
S4_TABLE = make_sbox_bit_table(s4_int)

def byte_match_relu(x_bits, pattern_bits):
    # For binary x, the agreement count is 8 only for the matching pattern.
    agreement = 0.0
    for x, p in zip(x_bits, pattern_bits):
        agreement += x if p == 1.0 else dnn_not(x)
    return relu(agreement - 7.0)

def dnn_sbox(x_bits, output_table):
    selectors = [
        byte_match_relu(x_bits, BYTE_PATTERNS[a])
        for a in range(256)
    ]
    output = []
    for output_bit in range(8):
        # Exactly one selector is active for a binary input.
        value = 0.0
        row = output_table[output_bit]
        for a in range(256):
            value += selectors[a] * row[a]
        output.append(value)
    return output


# ----------------------------------------------------------------------
# Camellia F, FL, and FL^-1 as ReLU networks
# ----------------------------------------------------------------------

def F_DNN(x_bits, k_bits):
    if len(x_bits) != 64 or len(k_bits) != 64:
        raise ValueError("F_DNN expects two 64-bit vectors.")

    u = xor_vec(x_bits, k_bits)
    b = [u[8*i:8*(i+1)] for i in range(8)]

    t1 = dnn_sbox(b[0], S1_TABLE)
    t2 = dnn_sbox(b[1], S2_TABLE)
    t3 = dnn_sbox(b[2], S3_TABLE)
    t4 = dnn_sbox(b[3], S4_TABLE)
    t5 = dnn_sbox(b[4], S2_TABLE)
    t6 = dnn_sbox(b[5], S3_TABLE)
    t7 = dnn_sbox(b[6], S4_TABLE)
    t8 = dnn_sbox(b[7], S1_TABLE)

    y1 = xor_many(t1, t3, t4, t6, t7, t8)
    y2 = xor_many(t1, t2, t4, t5, t7, t8)
    y3 = xor_many(t1, t2, t3, t5, t6, t8)
    y4 = xor_many(t2, t3, t4, t5, t6, t7)
    y5 = xor_many(t1, t2, t6, t7, t8)
    y6 = xor_many(t2, t3, t5, t7, t8)
    y7 = xor_many(t3, t4, t5, t6, t8)
    y8 = xor_many(t1, t4, t5, t6, t7)

    return y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8

def FL_DNN(x_bits, k_bits):
    xL, xR = x_bits[:32], x_bits[32:]
    kL, kR = k_bits[:32], k_bits[32:]

    yR = xor_vec(xR, rotl_bits(and_vec(xL, kL), 1))
    yL = xor_vec(xL, or_vec(yR, kR))
    return yL + yR

def FLINV_DNN(y_bits, k_bits):
    yL, yR = y_bits[:32], y_bits[32:]
    kL, kR = k_bits[:32], k_bits[32:]

    xL = xor_vec(yL, or_vec(yR, kR))
    xR = xor_vec(yR, rotl_bits(and_vec(xL, kL), 1))
    return xL + xR


# ----------------------------------------------------------------------
# Camellia-192 key schedule as a ReLU network
# ----------------------------------------------------------------------
# I extended your attached Camellia-128 ReLU-DNN implementation to Camellia-192, 
# while preserving the same overall functions and architecture. 
# Your original code realizes Boolean AND/XOR/OR/NOT exactly with ReLU and 
# implements the Camellia S-box through the 256-way ReLU selector network.
# The Camellia-192 version changes the key schedule to construct

# $$ K=K_L\Vert K_{RL},\qquad K_R=K_{RL}\Vert\overline{K_{RL}}, $$

# generates both \(K_A\) and \(K_B\), and uses 24 Feistel rounds with \(FL/FL^{-1}\)
# layers after rounds 6, 12, and 18. This follows the attached Camellia specification.

def generate_subkeys_192_DNN(key_bytes):
    if len(key_bytes) != 24:
        raise ValueError("Camellia-192 requires a 24-byte key.")

    # Specification: K = KL(128) || KRL(64), and KRR = complement(KRL).
    KL = bytes_to_bits(key_bytes[:16])
    KRL = bytes_to_bits(key_bytes[16:24])
    KRR = [dnn_not(x) for x in KRL]
    KR = KRL + KRR
    KLL, KLR = split128_bits(KL)

    D1 = xor_vec(KLL, KRL)
    D2 = xor_vec(KLR, KRR)
    D2 = xor_vec(D2, F_DNN(D1, int_to_bits(SIGMA[0], 64)))
    D1 = xor_vec(D1, F_DNN(D2, int_to_bits(SIGMA[1], 64)))
    D1 = xor_vec(D1, KLL)
    D2 = xor_vec(D2, KLR)
    D2 = xor_vec(D2, F_DNN(D1, int_to_bits(SIGMA[2], 64)))
    D1 = xor_vec(D1, F_DNN(D2, int_to_bits(SIGMA[3], 64)))
    KA = D1 + D2

    # KB = two further Feistel steps from KA XOR KR.
    D1, D2 = split128_bits(xor_vec(KA, KR))
    D2 = xor_vec(D2, F_DNN(D1, int_to_bits(SIGMA[4], 64)))
    D1 = xor_vec(D1, F_DNN(D2, int_to_bits(SIGMA[5], 64)))
    KB = D1 + D2

    def halves(x):
        return x[:64], x[64:]

    kw1, kw2 = halves(rotl_bits(KL, 0))
    k1, k2 = halves(rotl_bits(KB, 0))
    k3, k4 = halves(rotl_bits(KR, 15))
    k5, k6 = halves(rotl_bits(KA, 15))
    kl1, kl2 = halves(rotl_bits(KR, 30))
    k7, k8 = halves(rotl_bits(KB, 30))
    k9, k10 = halves(rotl_bits(KL, 45))
    k11, k12 = halves(rotl_bits(KA, 45))
    kl3, kl4 = halves(rotl_bits(KL, 60))
    k13, k14 = halves(rotl_bits(KR, 60))
    k15, k16 = halves(rotl_bits(KB, 60))
    k17, k18 = halves(rotl_bits(KL, 77))
    kl5, kl6 = halves(rotl_bits(KA, 77))
    k19, k20 = halves(rotl_bits(KR, 94))
    k21, k22 = halves(rotl_bits(KA, 94))
    k23, k24 = halves(rotl_bits(KL, 111))
    kw3, kw4 = halves(rotl_bits(KB, 111))

    return ([kw1, kw2, kw3, kw4],
            [k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,
             k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24],
            [kl1,kl2,kl3,kl4,kl5,kl6])


def camellia_encrypt_192_DNN(block_bytes, key_bytes):
    if len(block_bytes) != 16:
        raise ValueError("Camellia block size is 16 bytes.")
    kw, k, kl = generate_subkeys_192_DNN(key_bytes)
    state = bytes_to_bits(block_bytes)
    L, R = state[:64], state[64:]
    L, R = xor_vec(L, kw[0]), xor_vec(R, kw[1])
    for r in range(1, 25):
        L, R = xor_vec(R, F_DNN(L, k[r-1])), L
        if r == 6:
            L, R = FL_DNN(L, kl[0]), FLINV_DNN(R, kl[1])
        elif r == 12:
            L, R = FL_DNN(L, kl[2]), FLINV_DNN(R, kl[3])
        elif r == 18:
            L, R = FL_DNN(L, kl[4]), FLINV_DNN(R, kl[5])
    return bits_to_bytes(xor_vec(R, kw[2]) + xor_vec(L, kw[3]))


# ----------------------------------------------------------------------
# Independent Boolean reference implementation
# ----------------------------------------------------------------------

def F_BOOLEAN(x, k):
    x ^= k
    t = [(x >> (56 - 8*i)) & 0xFF for i in range(8)]
    t1,t2,t3,t4 = s1_int(t[0]),s2_int(t[1]),s3_int(t[2]),s4_int(t[3])
    t5,t6,t7,t8 = s2_int(t[4]),s3_int(t[5]),s4_int(t[6]),s1_int(t[7])
    y=[t1^t3^t4^t6^t7^t8,t1^t2^t4^t5^t7^t8,t1^t2^t3^t5^t6^t8,
       t2^t3^t4^t5^t6^t7,t1^t2^t6^t7^t8,t2^t3^t5^t7^t8,
       t3^t4^t5^t6^t8,t1^t4^t5^t6^t7]
    out=0
    for b in y: out=(out<<8)|b
    return out & MASK64

def rol32_int(x,n):
    n%=32; return ((x<<n)&MASK32)|((x&MASK32)>>(32-n))
def rol128_int(x,n):
    n%=128; return ((x<<n)&MASK128)|((x&MASK128)>>(128-n))
def FL_BOOLEAN(x,k):
    xL,xR=(x>>32)&MASK32,x&MASK32; kL,kR=(k>>32)&MASK32,k&MASK32
    xR ^= rol32_int(xL & kL,1); xL ^= xR | kR
    return ((xL&MASK32)<<32)|(xR&MASK32)
def FLINV_BOOLEAN(y,k):
    yL,yR=(y>>32)&MASK32,y&MASK32; kL,kR=(k>>32)&MASK32,k&MASK32
    yL ^= yR | kR; yR ^= rol32_int(yL & kL,1)
    return ((yL&MASK32)<<32)|(yR&MASK32)

def generate_subkeys_192_BOOLEAN(key_bytes):
    if len(key_bytes)!=24: raise ValueError("Camellia-192 requires a 24-byte key.")
    KL=int.from_bytes(key_bytes[:16],"big")
    KRL=int.from_bytes(key_bytes[16:],"big")
    KRR=(~KRL)&MASK64
    KR=((KRL&MASK64)<<64)|KRR
    KLL,KLR=(KL>>64)&MASK64,KL&MASK64
    D1,D2=KLL^KRL,KLR^KRR
    D2^=F_BOOLEAN(D1,SIGMA[0]); D1^=F_BOOLEAN(D2,SIGMA[1])
    D1^=KLL; D2^=KLR
    D2^=F_BOOLEAN(D1,SIGMA[2]); D1^=F_BOOLEAN(D2,SIGMA[3])
    KA=((D1&MASK64)<<64)|(D2&MASK64)
    X=KA^KR; D1,D2=(X>>64)&MASK64,X&MASK64
    D2^=F_BOOLEAN(D1,SIGMA[4]); D1^=F_BOOLEAN(D2,SIGMA[5])
    KB=((D1&MASK64)<<64)|(D2&MASK64)
    def halves(x): return (x>>64)&MASK64,x&MASK64
    kw1,kw2=halves(rol128_int(KL,0)); k1,k2=halves(rol128_int(KB,0))
    k3,k4=halves(rol128_int(KR,15)); k5,k6=halves(rol128_int(KA,15))
    kl1,kl2=halves(rol128_int(KR,30)); k7,k8=halves(rol128_int(KB,30))
    k9,k10=halves(rol128_int(KL,45)); k11,k12=halves(rol128_int(KA,45))
    kl3,kl4=halves(rol128_int(KL,60)); k13,k14=halves(rol128_int(KR,60))
    k15,k16=halves(rol128_int(KB,60)); k17,k18=halves(rol128_int(KL,77))
    kl5,kl6=halves(rol128_int(KA,77)); k19,k20=halves(rol128_int(KR,94))
    k21,k22=halves(rol128_int(KA,94)); k23,k24=halves(rol128_int(KL,111))
    kw3,kw4=halves(rol128_int(KB,111))
    return [kw1,kw2,kw3,kw4],[k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24],[kl1,kl2,kl3,kl4,kl5,kl6]

def camellia_encrypt_192_BOOLEAN(block_bytes,key_bytes):
    kw,k,kl=generate_subkeys_192_BOOLEAN(key_bytes)
    D=int.from_bytes(block_bytes,"big")^((kw[0]<<64)|kw[1])
    L,R=(D>>64)&MASK64,D&MASK64
    for r in range(1,25):
        L,R=(R^F_BOOLEAN(L,k[r-1]))&MASK64,L
        if r==6: L,R=FL_BOOLEAN(L,kl[0]),FLINV_BOOLEAN(R,kl[1])
        elif r==12: L,R=FL_BOOLEAN(L,kl[2]),FLINV_BOOLEAN(R,kl[3])
        elif r==18: L,R=FL_BOOLEAN(L,kl[4]),FLINV_BOOLEAN(R,kl[5])
    C=(((R&MASK64)<<64)|(L&MASK64))^((kw[2]<<64)|kw[3])
    return C.to_bytes(16,"big")


def verify_test_vector():
    # Official Camellia 192-bit test vector.
    pt=bytes.fromhex("0123456789abcdeffedcba9876543210")
    key=bytes.fromhex("0123456789abcdeffedcba98765432100011223344556677")
    expected=bytes.fromhex("b4993401b3e996f84ee5cee7d79b09b9")
    ct_boolean=camellia_encrypt_192_BOOLEAN(pt,key)
    ct_dnn=camellia_encrypt_192_DNN(pt,key)
    print("=== Camellia-192 Boolean vs ReLU-DNN Verification ===")
    print("PT           =",pt.hex()); print("KEY          =",key.hex())
    print("CT Boolean   =",ct_boolean.hex()); print("CT DNN       =",ct_dnn.hex())
    print("CT expected  =",expected.hex())
    print("Boolean==DNN =",ct_boolean==ct_dnn); print("Boolean OK   =",ct_boolean==expected); print("DNN OK       =",ct_dnn==expected)
    assert ct_boolean==expected
    assert ct_dnn==expected
    assert ct_boolean==ct_dnn
    print("VERIFICATION = PASS")


# ----------------------------------------------------------------------
# Camellia-192 external master-key recovery
# ----------------------------------------------------------------------

def rotr128_int(x, n):
    n %= 128
    return ((x >> n) | ((x << (128 - n)) & MASK128)) & MASK128


def recover_master_key_192_from_round_keys(round_keys):
    """
    Recover the 192-bit external key from Camellia-192 round keys.

        k3 || k4  = KR <<< 15
        k9 || k10 = KL <<< 45

    Hence:
        KR = ROTR128(k3 || k4, 15)
        KL = ROTR128(k9 || k10, 45)

    For Camellia-192:
        KR = R || ~R
        K  = KL || R
    """
    if len(round_keys) < 10:
        raise ValueError("At least round keys k1 through k10 are required.")

    # round_keys elements from generate_subkeys_192_DNN() are 64-bit bit vectors.
    k3 = bits_to_int(round_keys[2])
    k4 = bits_to_int(round_keys[3])
    k9 = bits_to_int(round_keys[8])
    k10 = bits_to_int(round_keys[9])

    KR_rot15 = ((k3 & MASK64) << 64) | (k4 & MASK64)
    KL_rot45 = ((k9 & MASK64) << 64) | (k10 & MASK64)

    KR = rotr128_int(KR_rot15, 15)
    KL = rotr128_int(KL_rot45, 45)

    R = (KR >> 64) & MASK64
    KR_right = KR & MASK64
    consistency_ok = (KR_right == ((~R) & MASK64))

    recovered_key = KL.to_bytes(16, "big") + R.to_bytes(8, "big")
    return KL, KR, recovered_key, consistency_ok


def master_key_recovery_demo_192(key_bytes=None):
    """
    Generate Camellia-192 DNN round keys and recover the original 192-bit key.
    If key_bytes is None, a fresh random 192-bit key is used.
    """
    import secrets

    if key_bytes is None:
        key_bytes = secrets.token_bytes(24)
    if len(key_bytes) != 24:
        raise ValueError("Camellia-192 requires a 24-byte key.")

    _, round_keys, _ = generate_subkeys_192_DNN(key_bytes)
    KL, KR, recovered_key, consistency_ok = \
        recover_master_key_192_from_round_keys(round_keys)

    print("=== Camellia-192 DNN Master-Key Recovery ===")
    print("Original key  =", key_bytes.hex())
    print("Recovered KL  =", f"{KL:032x}")
    print("Recovered KR  =", f"{KR:032x}")
    print("Recovered key =", recovered_key.hex())
    print("KR check      =", consistency_ok)
    print("Recovery      =", "SUCCESS" if recovered_key == key_bytes else "FAIL")

    assert consistency_ok, "KR = R || ~R consistency check failed."
    assert recovered_key == key_bytes, "Master-key recovery failed."
    return recovered_key


# ----------------------------------------------------------------------
# Repeated random-master-key recovery experiment (v2)
# ----------------------------------------------------------------------

def run_random_master_key_trials(num_trials=100):
    """Test recovery with a fresh random Camellia-192 master key each trial."""
    import secrets
    import time

    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")

    success = failure = consistency_failures = 0
    start = time.perf_counter()

    print("=" * 72)
    print("Camellia-192 Random Master-Key Recovery Experiment")
    print("=" * 72)
    print(f"Number of trials : {num_trials}\n")

    for trial in range(1, num_trials + 1):
        original_key = secrets.token_bytes(24)
        _, round_keys, _ = generate_subkeys_192_DNN(original_key)

        KL, KR, recovered_key, consistency_ok = \
            recover_master_key_192_from_round_keys(round_keys)

        recovered_ok = (recovered_key == original_key)

        if recovered_ok and consistency_ok:
            success += 1
            status = "SUCCESS"
        else:
            failure += 1
            status = "FAIL"
            if not consistency_ok:
                consistency_failures += 1


        if num_trials <= 100 or trial <= 10 or trial % 100 == 0 or status == "FAIL":
            print(f"Trial {trial:4d}/{num_trials}: {status}")
            print("Original key  =", original_key.hex())
            print("Recovered key =", recovered_key.hex())

        if status == "FAIL":
            print("  Original  =", original_key.hex())
            print("  Recovered =", recovered_key.hex())
            print("  KL        =", f"{KL:032x}")
            print("  KR        =", f"{KR:032x}")
            print("  KR check  =", consistency_ok)

    elapsed = time.perf_counter() - start
    rate = 100.0 * success / num_trials

    print("\n" + "=" * 72)
    print("Experiment Summary")
    print("=" * 72)
    print(f"Total trials             : {num_trials}")
    print(f"Successful recoveries    : {success}")
    print(f"Failed recoveries        : {failure}")
    print(f"KR consistency failures  : {consistency_failures}")
    print(f"Success rate             : {rate:.2f}%")
    print(f"Total elapsed time       : {elapsed:.3f} s")
    print(f"Average time / trial     : {elapsed / num_trials:.6f} s")
    print("=" * 72)

    return success, failure


if __name__ == "__main__":
    verify_test_vector()
    print()

    # Set the desired number of independent random-master-key trials here.
    # Examples: 20, 100, 1000
    NUM_TRIALS = 1000

    run_random_master_key_trials(NUM_TRIALS)
