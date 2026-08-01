# WHYS — why is Entropy 38.9% of JPEG encode?

Descent log. One entry per level. Never delete a level; a cause that turns out
small still explains why the fix aimed at it returned little.

Rule: **every "why" is answered by a measurement, never by a mechanism I can
explain.** A plausible explanation is the next hypothesis, not an answer.

---

## D6a — is the profile's content representative? **NO. Closes as a defect.**

- **ASKED:** before trusting "Entropy = 38.9%", does the benchmark content
  produce a realistic coefficient distribution?
- **MEASURED** (deterministic counters, immune to timing):
  - **39.1 non-zero AC coefficients per block, out of 63** — 62% of the AC
    spectrum is non-zero.
  - 40.6 Huffman symbols per block.
  - Whole-frame compression at q90: 12.4 MB raw → 5.01 MB = **2.5x**.
- **ANSWER:** the synthetic source (gradient + rings + per-pixel noise) is far
  denser than real imagery. A photograph at q90 compresses ~10-15x and leaves
  most of the AC spectrum zero. This content maximizes exactly the work Entropy
  does, so **38.9% is an upper bound peculiar to this clip, not the codec's
  profile.**
- **CONFIDENCE:** high — the count is exact and the compression ratio
  corroborates it independently.
- **CONSEQUENCE:** every share in the earlier breakdown is suspect in the same
  direction. Re-profile on content spanning realistic densities before choosing
  a target.
- **SPAWNED:** D6b (do the counters poison the cycles?), D3a (what is the share
  on realistic content?)
- **STATUS:** closed — instrument defect confirmed.

> This is the skill's own warning landing on me: the earlier guard test only
> asserted compression < 12x, which this content passes at 2.5x. A guard with a
> one-sided bound does not test for "representative", only for "not trivial".

## D6b — do the work counters poison the cycle measurement? **YES.**

- **ASKED:** the same profiled run reported `cycles/block 3289`, but the
  counter-free run reported 1074. Which is real?
- **MEASURED:** the counters fire ~558M times per run (138M symbols + 138M bit
  writes + 138M bit sums + 133M non-zero AC + 10.7M flushes), every one a
  relaxed `fetch_add` (`lock xadd`, ~20+ cycles), **all inside the Entropy
  scope**. 558M x ~20 = ~11 Gcycles, against a whole-run Total of ~9.2 Gcycles
  without them.
- **ANSWER:** with counters on, the Entropy bucket is measuring the instrument.
  The **counts are exact and usable; the cycles from that build are not.**
- **CONFIDENCE:** high — arithmetic reconciles with the observed 3x inflation.
- **ACTION:** counters moved behind their own `counters` feature, separate from
  `profile`, so cycles and counts are never read from the same binary.
- **STATUS:** closed.

---

## D3a — what is the share on REALISTIC content? **Entropy is 11.7%, not 38.9%.**

- **ASKED:** with a photographic fixture (1/f fractal detail, calibrated by
  sweep to 11.5x whole-frame compression), where does 4K encode actually go?
- **MEASURED** (profiler reset per density, so the buckets are not averaged
  across content classes — they were, initially, and the noise fixture swamped
  the photographic one):

  | stage | **photo (11.5x)** | noise (2.0x) |
  |---|---|---|
  | **Quantize** | **24.40%** | 21.86% |
  | FillBuffers | 17.31% | 10.68% |
  | residue | 15.14% | 10.29% |
  | **GetBlock** | **13.12%** | 7.80% |
  | **Entropy** | **11.69%** | **28.87%** |
  | HuffmanOptimize | 10.92% | 13.53% |
  | Fdct | 7.42% | 6.96% |

- **ANSWER:** Entropy's dominance was an artifact of the fixture. On content a
  user would actually encode it is **fifth**, at 11.7%. The real leader is
  **Quantize at 24.4%**, with FillBuffers + GetBlock (the block-assembly glue)
  together at **30.4%**.
- **CONFIDENCE:** high — the two profiles come from the same binary and differ
  only in input; the direction matches the counter evidence (50 vs 6.6 non-zero
  AC per block) exactly as predicted.
- **CONSEQUENCE:** **the stated goal of this session — "hammer Entropy" — was
  aimed by a broken instrument.** Optimizing Entropy to zero would buy 11.7% on
  realistic content, not 38.9%. Amdahl caps it there.
- **SPAWNED:** D4a (why is Quantize 24%?), D4b (why is block assembly 30%?)
- **STATUS:** closed.

## D4a — is Quantize's sign branch a mispredict? **NO. REFUTED.**

- **ASKED:** `quantize()` ends with `if value != abs_value { product *= -1 }`.
  DCT signs are essentially random, so that reads as a coin-flip branch run 64x
  per block — ~174M unpredictable branches for one 4K frame. Is it?
- **MEASURED:**
  - Hand-branchless rewrite (`(v ^ sign) - sign`), proven bit-identical
    **exhaustively** — all 65536 `i16` values x 64 positions x 6 qualities x
    luma/chroma.
  - Paired ABBA A/B, n=25, both content classes, arms differing *only* in the
    sign fixup:

    | arm | wins | z | median B/A |
    |---|---|---|---|
    | null [photo] | 14/25 | 0.60 | 1.0483 |
    | branchy [photo] | 10/25 | **-1.00** | 0.9456 |
    | null [noise] | 12/25 | -0.20 | 1.0000 |
    | branchy [noise] | 8/25 | **-1.80** | 0.9685 |

  - `--emit asm`: **156 `cmov` instructions** in the crate.
- **ANSWER:** LLVM already compiles that pattern to a conditional move. There
  was no branch to remove; the rewrite swapped one branchless form for another
  and measured *marginally worse* on both content classes.
- **CONFIDENCE:** high — two content classes agree in sign, the null arms are
  clean, and the `cmov` count gives the mechanism rather than just a null result.
- **ACTION:** shipped path **reverted** to the original. Both forms and the
  `set_branchy_quantize` knob retained so the re-test is free if the surrounding
  stages shrink or a target compiler stops emitting the move.
- **SCOPE OF THE REFUTATION:** this kills *"the sign fixup is a mispredicting
  branch worth removing by hand"*. It does **not** kill *"Quantize at ~24% has
  headroom"* — the ZIGZAG permutation gather is untested and still open.
- **STATUS:** closed, refuted.

## D4b — why is FillBuffers 17.3%? **Per-byte `push`. CONFIRMED, FIXED.**

- **ASKED:** for planar input, `fill_buffers` should be close to a memcpy. Why
  is it a sixth of encode?
- **MEASURED:** the chroma path was
  `for &c in src { for _ in 0..sh { dst.push(c) } }` — a capacity check and a
  length update **per byte**, ~16.6M of them for one 4K frame. (My own code,
  added earlier this session.)
- **FIX:** size the destination once (`resize`), then write through a slice with
  a constant-width chunk for the common `sh == 2` case. Byte-identical; the
  exact-replication tests, including the ragged-width one, still pass.
- **GATE** — paired ABBA, whole encode (the level *above* the change):

  | arm | wins | z | median B/A |
  |---|---|---|---|
  | null [4K] | 19/41 | -0.47 | 0.9898 |
  | push [4K] | **37/41** | **+5.15** | **1.2547** |

- **STAGE CONFIRMATION:** FillBuffers **17.31% -> 1.02%** of encode.
- **ANSWER:** confirmed and shipped. ~**1.25x whole-encode** at 4K.
- **ON THE MAGNITUDE:** at n=25 the median read 1.3226, which *exceeds* what
  Amdahl allows for a 17.3% stage (1.21x cap) — an impossible number, and the
  tell that the estimator was under-sampled. At n=41 it settled to 1.2547,
  consistent with the ~1.20x the share arithmetic predicts. **Win-rate was
  decisive at n=25; the magnitude was not.** Quote the range, not the
  flattering end.
- **STATUS:** closed, shipped.

## D4c — what is the residue? **Mostly the instrument. Instrument-limited.**

- **ASKED:** residue was the largest line (24-28%). Real work, or the probe?
- **MEASURED:** stopped assuming the probe cost and **measured** it — an empty
  `scope()` in a tight loop costs **~88 cycles**, not the ~40 the dump had been
  assuming. At 6.83M scope entries that is 605 Mcycles = **15.5% of Total**.
- **BISECTED:** an info-tier `BlockBody` scope (excluded from sums, so nesting
  cannot drive the residue negative — it did, to -165%, when the first version
  of these scopes overlapped their children) put ~509 Mcycles "inside the block
  loop". Of that, 3 nested probes x 1.36M x 88 = ~357 Mcycles is probe.
- **TWO BRICKS AIMED AT THE REMAINDER, BOTH REFUTED:**
  - in-place block write vs `push` of a stack temporary: **20/41, z = -0.16,
    median 0.9996** — LLVM already elides that copy.
  - (see D4a) the same story for the sign fixup.
  Both are explained by the same fact: there was far less glue than the raw
  residue implied.
- **ANSWER:** the residue is dominated by measurement cost at this scope
  granularity. Correcting per-stage makes `Fdct` read **11 cycles/block** for an
  AVX2 8x8 DCT — impossible, so the empty-loop figure is an UPPER bound (in a
  real loop the two `lock xadd`s partly overlap with surrounding work) and the
  true residue sits in a band this granularity cannot resolve.
- **CONFIDENCE:** high on the conclusion, low on any single number — which is
  the point.
- **ACTION:** `dump()` now measures the probe cost and prints a probe-corrected
  table plus the caveat. **The fix for the ambiguity is fewer scopes, not a
  better estimate**: coarsen the per-block scopes to per-block-row (1000x fewer
  entries) and the correction becomes negligible.
- **STATUS:** closed — instrument-limited, not a code target.

## D4d — can Quantize be cut? *(three refutations, one lever left)*

- Pre-permuting `reciprocals`/`corrections` into zig-zag order, so two of the
  three permuted reads per coefficient become sequential (proven bit-identical
  exhaustively): **12/25 z = -0.20 [photo], 8/25 z = -1.80 [noise] — inside
  noise.** No win. The indices were compile-time constants in a fully unrolled
  loop, so there was no indexing cost to remove.
- With D4a (sign branch) and D4c (block copy), that is **three refuted
  hypotheses on this kernel**. The pattern is consistent: LLVM has already
  optimized this loop hard.
- **`--emit asm` confirms partial auto-vectorization already present:**
  `vpmulld` 25, `vpaddd` 52, `vpsrad` 12 (alongside 176 scalar `imul`).
- **REMAINING LEVER:** hand-AVX2 quantize (8 coefficients per iteration,
  `_mm256_sign_epi32` for the sign restore). It must beat an *already partly
  vectorized* loop, so it is not the easy win it looked like.
- **CEILING, priced before building:** Quantize is ~19.8% probe-corrected.
  Halving it buys ~10% of encode (~1.11x). Worth doing, not transformative.
- **PREREQUISITE:** `quantize(0, i) == 0` must hold at every position, or
  `_mm256_sign_epi32` (which zeroes when the sign operand is zero) is not a
  valid substitute for the scalar sign fixup. Check before writing intrinsics.
- **STATUS:** closed — see D5a.

## D5a — hand-AVX2 quantize. **CONFIRMED. Shipped. Stage cut ~58%.**

Five probes, per the deepened three-probe rule.

**Probe 1 — does it auto-vectorize?** Pulled the loop into an `#[inline(never)]`
`quantize_block_scalar` purely so it would have a symbol to disassemble, then
read the asm. A 64-iteration scalar loop, **zero packed instructions**. It also
settled D4a for good: `cmovsw`/`cmovnsl` — the sign fixup was branchless all
along — and confirmed D4d, since the two table reads index by `%r9` (`i`,
sequential) while only `movzwl (%rcx,%r10,2)` indexes by `%r10` (`ZIGZAG[i]`).

> **The named blocker: one gather.** LLVM will not synthesise a 64-element
> cross-lane permutation, so a single permuted load kept the entire loop scalar.

**Probe 2 — preconditions.** The kernel leans on two instructions whose
semantics differ from the scalar code, so both were checked, not assumed:
- `_mm256_sign_epi32(q, v)` returns **0** when `v == 0`, while the scalar path
  computes `((0 + corr) * recip) >> SHIFT`. Verified equal across **all 9 table
  types x 100 qualities x luma/chroma x 64 positions**.
- `_mm256_packs_epi32` **saturates**; `product as i16` **truncates**. Verified
  quantized magnitudes stay inside i16.
- Overflow bound by hand: `abs <= 32767`, `recip = 32768/divisor <= 4096`
  (divisors are stored pre-shifted, minimum 8), so the product tops out ~1.34e8,
  inside i32.

**The fix follows from the diagnosis**, rather than being a generic "add SIMD":
quantize in **natural order** — where block, reciprocals and corrections are all
sequential and the loop vectorizes — then apply the zig-zag permutation once
afterwards as pure data movement.

**Probe 3 — scalar oracle gate.** `assert_eq!` (integer kernel, no tolerance)
over 4 table types x 5 qualities x luma/chroma x 64 blocks, including the edge
cases that matter: all-zero, all-`i16::MAX`, all-`i16::MIN`, alternating
extremes. Bit-identical.

**Probe 4 — in-context paired A/B**, arms differing only in the quantizer
(AVX2 FDCT on both sides, so the comparison is single-variable):

| arm | wins | z | median B/A |
|---|---|---|---|
| null [photo] | 9/25 | -1.40 | 0.9755 |
| **scalar [photo]** | **25/25** | **+5.00** | **1.1513** |
| null [noise] | 16/25 | 1.40 | 1.0162 |
| scalar [noise] | 22/25 | +3.80 | 1.0361 |

**Probe 5 — end-to-end byte identity.** The kernel is bit-identical, so the
emitted file must be too: verified byte-for-byte at 64x64, **127x65** (odd
dimensions, exercising the ragged final block) and 1920x1080, on both content
classes. The first run of this gate failed — on the **test fixture**, which
sized chroma with `w/2` instead of `ceil(w/2)`, and `PlanarYcbcrImage::new`
correctly rejected the under-sized planes. Fixture bug, not kernel bug.

**RESULT:** Quantize **15.64% -> 7.48%** of encode; combined with the encode
itself getting 1.15x faster, the stage is now ~42% of its former cost — **cut by
~58%**. Whole encode **1.15x** on photographic content.

**End-to-end vs FFmpeg (4K y4m -> mjpeg, whole CLI, best-of-9, same I/O both
sides): 1.51x -> 1.06x.**

---

# Decoder descent

## D6e — do both arms decode the SAME BITSTREAM? **NO. The 2.20x was void.**

- **ASKED** (codec-measurement §4, work parity): both arms report a byte count.
  Do they match?
- **MEASURED:** they did not, and by a lot.
  - we decoded `photo.jpg` — **261,455 B**
  - ffmpeg decoded a frame of `photo300.avi` — **139,963 B**
  - I had built the reference clip with `-c:v mjpeg -q:v 3`, which **re-encoded**
    it. ffmpeg was decoding a stream with roughly **half the coefficients**.
- **FIX:** rebuilt with `-c:v copy` and asserted the extracted frame is
  byte-identical to `photo.jpg` before measuring anything.
- **RE-MEASURED** (300 frames, identical bitstream, pinned CPU time,
  single-thread both sides): ffmpeg **6.20 ms/frame**, ours **9.38** —
  **1.51x slower, not 2.20x.**
- **CONFIDENCE:** high — parity asserted by `cmp`, spreads 1.05-1.06x.
- **LESSON:** this is the third flattering-or-damning measurement in this
  campaign killed by a work-parity check, and the cheapest one to have run. The
  size of the two files differed by 1.87x and was visible the whole time.
- **STATUS:** closed.

## D3c — FUNCTION-LEVEL split by ABLATION, not by scopes. **Entropy is 41.9%.**

- **ASKED:** the scope-based profile said entropy ~11%, IDCT ~10%, plumbing
  ~78%. Those stages are entered 2.9M times each, and codec-measurement §6 says
  a share is only trustworthy when the CALL COUNT is small — high-call stages
  must be priced by **ablation on the uninstrumented binary**.
- **MEASURED** (300 x 1080p photo, pinned CPU time, medians; the ablations keep
  every surrounding access and delete only the work):

  | arm | cpu_med |
  |---|---|
  | baseline | 3093.8 ms |
  | `ABLATE=idct` | 2781.3 ms |
  | `ABLATE=entropy` | 1921.9 ms |
  | `ABLATE=idct,entropy` | 1484.4 ms |

  Solving (`E` = entropy, `I` = transform above the stores, `R` = everything else):
  - `R` = 1484.4
  - `E` = 2781.3 - 1484.4 = **1296.9 ms = 41.9%**
  - `I` = 3093.8 - 2781.3 = **312.5 ms = 10.1%**
  - identity checks: 1296.9 + 312.5 + 1484.4 = 3093.8 exactly.

- **ANSWER:** **entropy decode is 41.9% of decode, not 11%.** The scope-based
  figure was inflated away from it by the probe, in the direction §6 predicts:
  per-block scopes charge their own tax into the stage AND the residue, so the
  real hot stage looked small and the "plumbing residue" looked huge.
- **CONSEQUENCE — this reopens a prune.** I previously declined the bit-reader
  rewrite by computing its prize as ~8%, using the probe-distorted 25% share.
  On the ablated 41.9%, **making entropy free is worth 41.9%**, and the target
  needs 40.5% removed. The arithmetic that said "cannot close" was computed on a
  bad number. **codec-measurement §11: a refutation expires when its baseline
  moves — and mine moved because the baseline was wrong.**
- **CAVEAT, stated rather than buried:** the IDCT figure is NOT solid — the
  medians say 312.5 ms but the minima invert (baseline 2562.5 vs no-IDCT
  2578.1), i.e. the effect sits inside this run's spread (1.11-1.38x). Entropy's
  1296.9 ms is far outside it and is the number to act on.
- **STATUS:** closed for entropy; IDCT share open and under-resolved.

## D4e — what inside entropy? **260,265 `read_exact` calls per frame.**

- **MEASURED** (deterministic counters, 1080p photographic frame):

  | unit | per frame |
  |---|---|
  | **bytes read** | **260,265** |
  | refills | 38,378 |
  | symbols | 126,846 |
  | receive_extend | 67,491 |

- **ANSWER:** `bytes_read` is essentially the whole 261,455 B file. The Huffman
  decoder is already LUT-based with a 64-bit accumulator — the cost is not the
  symbol decode, it is that the accumulator refills **one byte at a time**
  through `read_u8` -> `Cursor::read_exact`, ~6.8 times per refill. Against the
  ablated entropy cost that is roughly **50 cycles per byte** for what should be
  a bounds check and an index.
- **STATUS:** closed.

## D5c — shared buffered reader. **CONFIRMED, SHIPPED. 1.14x whole-decode.**

- The buffer has to live at the **decoder** level, not inside the Huffman
  decoder: reading ahead would swallow bytes the marker and segment parsers need
  next, and a bare `Read` cannot push them back. Because `Buffered<R>` itself
  implements `Read`, every existing generic call site (`parse_sof`, `parse_dht`,
  `parse_app`, …) keeps working unchanged and shares the same buffer; only the
  entropy path takes the concrete type so it can use `next_byte`.
- **GATE:** 55 tests pass, and decoded output is checksum-identical to the
  byte-at-a-time path on four images across both planar and RGB paths.
- **A/B** — paired ABBA over separate processes, pinned CPU time (sequential
  block-vs-block was useless here: the reference itself moved 5% between runs):

  | arm | A-wins | z | median B/A |
  |---|---|---|---|
  | NULL (buffered vs itself) | 8/21 | -1.09 | 0.9900 |
  | **buffered vs `read_u8`** | **21/21** | **+4.58** | **1.1376** |

- **A defect in my own instrument, caught and fixed:** the ablation knob was
  first placed *inside* the byte loop, putting a `OnceLock` atomic load on the
  hottest path in the decoder — 260k times per frame — to serve a switch that
  never changes. Hoisted to once per refill.
- **STATUS:** closed, shipped.

### Standing, and which number is which

- **Progress (single instrument, paired, trustworthy): 1.14x.**
- **Standing vs FFmpeg (cross-implementation ratio, unstable denominator):**
  ~1.5-1.65x slower. FFmpeg's own measured decode moved 1859 -> 1688 ms between
  two runs of the same command in the same session, which is larger than the
  improvement being measured — so per codec-measurement §12 this ratio is
  reported for standing only and never as progress.

## D6c — is the decoder comparison fair? **NO, twice.** Both errors flattered us.

- **ASKED:** unpinned wall said our 1080p decode was 3.65 ms/frame against
  ffmpeg's 4.28 (`-threads 1`) — 1.17x faster. Is that real?
- **MEASURED:** it is not.
  1. **Thread count.** With the `rayon` feature off, `select_worker` still picks
     a **std::thread mpsc worker** for anything above 128x128. So our arm was
     using several cores while ffmpeg was pinned to one. Different work, not
     less work.
  2. **Wall vs CPU on a migrating process.** Three identical in-process runs read
     3.59 / 4.85 / 3.92 ms — a **35% spread**. Pinning to one core at High
     priority and measuring **CPU time** dropped the spread to **1.03-1.08x**.
- **THE HONEST NUMBER** (300 x 1080p 4:2:0, pinned, CPU time, single-thread both
  sides): ffmpeg **3.70 ms/frame**, ours **7.86 ms/frame** — we are **~2.1x
  SLOWER in real CPU work.** Threaded and single-threaded are within 1% of each
  other when pinned, so the earlier "faster" reading was entirely extra cores.
- **STATUS:** closed. Two flattering numbers, both from unfair comparisons; the
  second was found only because the first was re-checked under pinning.

## D3b — where does decode time actually go? **Not the kernels.**

Probe-corrected (`DecEntropy` and `DecIdct` have identical call counts, so their
probe cost cancels and their ratio is unbiased):

| stage | real share |
|---|---|
| DecEntropy | ~11% |
| DecIdct | ~10% |
| **everything else** | **~78%** |

- **ANSWER:** entropy decode and the IDCT together are barely a fifth of our
  decode. The 2.1x gap is **not** in the kernels — SSSE3 IDCT is wired and does
  get called. It is in the plumbing around them.
- **FIRST NAMED CAUSE, FIXED:** `decode_scan` allocated and zeroed a fresh
  `Vec<i16>` for **every MCU row of every component** — ~6.3 MB per 1080p frame.
  Added `Worker::reclaim_buffer`, which the synchronous immediate worker
  implements by handing its finished buffer back; threaded workers keep the
  allocating fallback. **2562 -> 2359 ms (~1.09x).**
- **SECOND SUSPECT, REFUTED:** hoisting the per-block
  `is_x86_feature_detected!` + indirect IDCT call out to once-per-scan
  (~14.6M feature checks removed) measured **neutral to slightly worse**
  (median 2453 vs 2359 ms; identical minima). The macro already caches into a
  static, so the check was a predicted-taken branch on a hot atomic. Reverted —
  it added an `unsafe` block for no measured gain.

### Where the remaining ~2.2x actually is

Coarse scopes (low call count, so probe-free) bisect `decode_scan`, which
contains **99.6%** of decode:

| stage | share | calls |
|---|---|---|
| DecEntropy | 28.3% | 2.94M |
| DecIdct | 25.3% | 2.94M |
| **DecPlaneInit** | **12.1%** | **180** |
| DecOutput | 0.02% | 60 |
| scan glue | ~34% | — |

- **DecPlaneInit** is 751K cycles per call to allocate and zero one output
  plane. Every byte is then overwritten by the IDCT, so the zeroing is pure
  waste — but the buffer *becomes* the returned image, so it cannot simply be
  reused. The cost profile (2 MB plane, ~512 page faults) says this is
  **page-fault cost on fresh allocation**, and the fix is a caller-supplied
  buffer pool, i.e. an API change up through `rff-codec-jpeg`.
- **DecEntropy** is 108 cycles/block. The bit reader accumulates into a `u64`
  but refills **one byte at a time** through `std::io::Read`, branching on
  `0xFF` per byte. FFmpeg unescapes the entropy stream up front and then does
  unbranched 64-bit loads. That is a structural difference, not a tuning gap.
- **DecIdct** is 97 cycles/block against an SSSE3 kernel; FFmpeg's is AVX2.

**Honest arithmetic:** perfect execution on all four lines is worth roughly
1.5-1.6x. The goal (10% faster than FFmpeg) needs **~2.4x**. It is a decoder
rewrite campaign — bulk bit reader + AVX2 IDCT + buffer pooling — not a set of
bricks.
- **STATUS:** open, correctly aimed, not met.

## D6d — is the DECODE benchmark content representative? **NO. Same trap as D6a.**

- **MEASURED (deterministic counter, DC-only block fraction):**
  `testsrc2` **92.6%**, noise **0.0%**, every other JPEG in the scratch tree
  0.0-0.6%. A DC-only block skips the entire 8x8 inverse transform, so those two
  extremes exercise two different decoders. **I had no photographic decode
  content at all.**
- **ACTION:** added `examples/make_fixture.rs`, which writes a 1/f fractal
  fixture at the amplitude calibrated for the encoder work — 11.9x compression,
  **31.6% DC-only**, i.e. genuinely photographic.
- **CONSEQUENCE — and the hypothesis this killed:** I expected the gap to be an
  artifact of ffmpeg having a DC-only fast path that we lacked. It is not. The
  ratio is **2.22x on photographic content and 2.2x on 92.6%-DC content** — the
  same. FFmpeg's advantage is uniform across content, so it is structural
  per-block work, not a missing special case.
- **STATUS:** closed. Second time in this campaign that synthetic fixture
  content nearly aimed a campaign at the wrong thing.

## D5b — DC-only IDCT shortcut. Bit-identical, strictly less work, ~neutral.

- Neither the scalar nor the SSSE3 IDCT had a whole-block DC-only path, though
  the scalar one already short-circuits per *column*. Added one **ahead of the
  arch dispatch**, so the vectorized path benefits too.
- **Derived, not approximated:** the column pass writes
  `dcterm = dequantize(c0,q0) << 2` down column 0 and zero elsewhere, so every
  row hits the existing `rest == [0;7]` branch with the same `s0`. Gated
  exhaustively over the whole i16 DC range x 9 quantizer values, all 64 output
  samples (`dc_only_matches_full_idct`), plus a test that a non-zero AC at any of
  the 63 positions does **not** take it.
- **MEASURED:** photo 3125 -> 3125 ms median (min 3000 -> 2875); flat
  2516 -> 2469 ms. **Inside noise.** The predicted ~8% did not appear because
  DecIdct's 25% share was probe-inflated; its real share is smaller.
- **KEPT** on "bit-identical and strictly less work", not on a speed claim: the
  `all(|c| c == 0)` scan short-circuits on the first non-zero, so content where
  it never fires pays one load and one compare. It matters most on the
  `platform_independent` build, which has no SSSE3 kernel at all.

---

## D6f - the "single-threaded" arm was multithreaded the whole time

- **ASKED:** `DecRowDispatch` measured 65.6 Mcycles while `DecIdct`, which nests
  INSIDE it, measured 204.7. A parent cannot be cheaper than its child. Which is
  lying, the profiler or the placement?
- **MEASURED:**
  - A nesting unit test (`parent_scope_is_never_smaller_than_its_child`) passes
    -> the profiler is sound.
  - Only one `DecIdct` call site exists (`immediate.rs`), inside `append_row`.
  - `st` and non-`st` runs produced *indistinguishable* profiles.
  - A trace of the chosen worker printed `Multithreaded` for BOTH.
- **ANSWER:** `decoder.rs` hardcoded `PreferWorkerKind::Multithreaded` at the
  scan-loop call site. `get_or_init_worker` CACHES the worker on first use, and
  for a baseline image that site runs before the one in `decode_planes` that
  honours the flag. So `set_single_threaded` was a **public API that did
  nothing** on baseline JPEGs. The inversion was real: `append_row` only queued,
  and the IDCT ran on the worker thread, outside the scope.
- **CONSEQUENCES**, all of which had been silently priced wrong:
  - Every "single-threaded" decode measurement in this campaign, including the
    comparison against `ffmpeg -threads 1`, was a multithreaded arm. The
    `cpu/wall < 1` check from `codec-measurement` Sec.2 had never actually passed.
  - `reclaim_buffer` -- the MCU-row buffer recycling shipped earlier -- is
    implemented ONLY by the immediate worker. It was dead code in the real path,
    so every MCU row reallocated ~6.3 MB/frame. With the flag live,
    `DecMcuRowAlloc` fell 5.82% -> **1.57%**.
- **WORTH:** on one pinned core, genuinely single-threaded decode costs
  **2015.6 ms vs 2781.3 ms** for 300 frames -- the threaded path burned **38%
  more CPU**. Unpinned it is still the right default (6.97 vs 7.96 ms/frame wall,
  1.14x), so the default stands and only the flag was repaired.
- **GATED:** `single_threaded_flag_selects_the_immediate_worker_on_baseline`
  asserts both the worker choice and that output is unchanged. The choice is now
  observable via `decode::last_worker_was_immediate()`, so this cannot silently
  regress again.
- **CONFIDENCE:** high. Checksums identical across both workers; full suite green.

## D6g - the plane-init "66.7% of decode" was a crash

- **ASKED:** plane recycling was measured at 1.086x (21/21, z=+4.58) but the
  ablation had attributed **66.7%** of decode to plane initialization. A ~10x
  disagreement.
- **MEASURED:** the `planeinit` ablation arm exits **101** -- it panics
  (`range start index 8 out of range for slice of length 0`). Its "15.6 ms for
  300 1080p decodes" was the process dying, and the harness counted it as data.
- **ANSWER:** there is no 66.7%. The honest figure is the gated A/B: ~8%.
  0.05 ms/frame for a 1080p decode was impossible on its face -- exactly the
  Sec.7 tell, and it went unchallenged for a whole peel.
- **GATED:** `pinned.ps1` and `paired.ps1` now `throw` on any non-zero exit. A
  crashed run can never enter a median again.

## D6h - the decode fixture was non-interleaved, and so is our encoder's output

- **ASKED:** `DecBlockLoop` fired once per BLOCK rather than once per MCU. Why?
- **MEASURED:** `photo.jpg` has **three single-component scans**. Sweeping every
  fixture: everything ffmpeg wrote is interleaved; everything our encoder wrote
  with `optimize_huffman` on is **non-interleaved**.
- **ANSWER:** `use_streaming_optimize` picks the encode route on a **memory
  budget** (`OPTIMIZE_BUFFER_BUDGET = 256 MB`). Under it, `encode_image_sequential`
  runs and writes one scan PER COMPONENT. At 1080p that is 12.5 MB and at 4K
  50 MB, so at every practical resolution an ordinary `-optimize` request yields
  a layout mainstream encoders never emit. **The scan layout of our output
  depends on how much RAM the encoder felt like using.**
- **STATUS:** open (encoder-side fix). Forcing the streaming route gives
  interleaved output at byte-equivalent size (263,657 vs 262,145 B) for ~23% more
  encode time; making the materializing route write MCU-order would avoid that
  cost but needs the histogram gathered in the same order.
- **SPAWNED:** D6i.

## D6i - the ffmpeg decode arm was decoding a different, smaller file

- **ASKED:** re-baselining on interleaved content, does the reference arm still
  do identical work?
- **MEASURED:** `dec300.avi` is 30 MB / 300 = 100 KB per frame. `photo.jpg` is
  **261 KB**. Extracting frame 0 with `-c:v copy` and `cmp`-ing: it is
  **`one.jpg`**, not `photo.jpg`.
- **ANSWER:** the reference decoded **2.67x less entropy data** than we did, and
  an interleaved stream against our non-interleaved one. Every decoder ratio
  taken through that harness is **void** -- this is the second occurrence of the
  same work-parity failure (see D6e).
- **REBUILT:** clips are now muxed `-c:v copy` from the exact fixture and gated
  by `cmp` against the source before any timing runs.
- **STANDING, on representative interleaved content, both arms verified:**

  | arm | cpu_med | ms/frame |
  |---|---:|---:|
  | ours, single-threaded | 2015.6 | 6.72 |
  | ffmpeg -threads 1, net of demux | 1812.5 | 6.04 |

  **~1.10x slower** -- not the 1.47x, and not the 1.63x the broken arms implied.

## D6j - the encoder premise, verified like-for-like

- **ASKED:** the standing goal is conditioned on "the encoder is 15% faster than
  ffmpeg". That number came from the same harness that produced four void
  decoder measurements. Does it survive?
- **MEASURED:** at equal `-q:v` our output is **41% larger** -- not the same
  operating point, so equal-flag timing is meaningless. Matching on OUTPUT SIZE
  instead, and equalising the Huffman configuration:

  | arm | output | net encode (40x 1080p) |
  |---|---:|---:|
  | ffmpeg `-q:v 3` | 28.04 MB | 1140.6 ms |
  | ours `-q:v 5 -optimize_huffman 0` | 27.17 MB | **937.5 ms** |
  | ours `-q:v 4 -optimize_huffman 0` | 32.68 MB | 1078.1 ms |

  Interpolated to ffmpeg's exact size: ~960 ms -> **1.19x faster (16%)**.
- **ANSWER:** the premise **holds**. But only like-for-like: our CLI defaults
  `optimize_huffman = true` (two-pass, and via the D6h memory-budget gate the
  block-MATERIALIZING two-pass, ~12.5 MB/frame) while ffmpeg's mjpeg encoder
  defaults to fixed tables in one pass. Compared on DEFAULTS we read ~1.37x
  slower -- doing more work for smaller output, which is a different tradeoff,
  not a speed deficit. Sec.8 again: the reference's defaults are configuration,
  and so are ours.
- **CONFIDENCE:** high. Both arms pinned, CPU time, null arm subtracted, sizes
  within 3.1%.

## D3d - entropy is 52.4%, but the obvious lever is arithmetically dead

- **MEASURED (deterministic counters, per 1080p frame):**
  `lut_hit` 127,271 vs `lut_MISS` **2,062** -> the primary Huffman LUT misses
  **1.6%** of the time. Widening `LUT_BITS` cannot pay for itself, and would cost
  L1. **Refuted on arithmetic before building anything** (Sec.11).
  `decode_fast_ac` is a different story: 281,629 hits vs **80,373 misses (22%)**,
  each falling back to a full `decode` + `receive_extend` that re-peeks the same
  bits.
- **TRIED:** hoisted the two loop-invariants off the ~362k-calls/frame fast-AC
  path -- the `Option` discriminant on `ac_lut` (now an always-present zero table
  for DC) and `ac_table.unwrap()`.
- **MEASURED:** 1968.8 -> 1953.1 ms, with ffmpeg re-run as a drift anchor in the
  same session (1890.6 -> 1875.0, so the box held). 0.8% is **inside the noise
  floor**. LLVM was already hoisting it.
- **KEPT** on "strictly less work, no correctness risk, simpler type", NOT on a
  speed claim -- recorded as an inside-noise result, not a win (Sec.12).
- **NEXT LEVER, with its arithmetic:** IDCT is 24.6% and runs SSSE3 while the box
  has AVX2. Widening the existing kernel to two blocks per instruction stream is
  bit-exact by construction (AVX2 integer ops and the unpack transpose are
  lane-independent). Even at 1.5x on the stage that is only **1.089x** overall,
  so it is necessary but not sufficient for the 1.235x the goal needs.

## D5d - AVX2 two-block IDCT: SHIPPED, 7.8% whole-decode

- **PRIZE, computed before building (Sec.11):** IDCT 24.6% of decode, kernel is
  SSSE3 on a box with AVX2. Two blocks per instruction stream -> at 1.5x on the
  stage, 1.089x overall. Necessary, not sufficient; built anyway because it was
  the largest single lever left.
- **WHY IT IS SAFE:** block A in the low 128-bit lane, block B in the high one.
  Every op involved is lane-independent -- the arithmetic is elementwise, and
  `_mm256_unpack*` / `_mm256_packus_epi16` never cross the lane boundary, so the
  8x8 transpose transposes each block separately for free. Each lane therefore
  executes exactly the SSSE3 sequence on exactly the same inputs.
- **GATED:** `avx2_pair_matches_ssse3_twice` asserts **byte-identical** output vs
  calling the SSSE3 kernel twice, over 64 rounds including an `i16::MAX` /
  `i16::MIN` / `u16::MAX` saturation round. Not a tolerance -- if the two
  disagreed at all, decoder output would depend on which CPU ran it.
  Whole-image checksums unchanged on all five fixtures; suite 58/58.
- **COVERAGE, proven at runtime (Sec.10):** `idct_PAIRS` = 16,617/frame = 33,234
  blocks, i.e. **98.5%** of the 33,732 blocks that need a full transform. The
  other 31.1% are DC-only and take a cheaper fill.
- **MEASURED:** 1953.1 -> 1812.5 ms = **7.8%**, against a drift anchor that moved
  the WRONG way in the same session (ffmpeg 1875.0 -> 1890.6), so the gain is not
  the box getting faster.
- **TWO SELF-INFLICTED BUGS THE PEEL THEN CAUGHT:**
  1. `-idct` ablation read HIGHER than the un-ablated run. The knob lives inside
     `dequantize_and_idct_block`, which the pair path bypasses -- a **stale knob**
     that had silently stopped measuring the stage it names. Now also gates the
     pair path.
  2. `-entropy` went 937.5 -> 1234.4 ms. With entropy ablated every block is
     DC-only, which exposed that pairing made DC-only blocks scan `is_dc_only`
     **twice** -- once at the call site, once inside. Split out `fill_dc_only` so
     a caller that already knows takes the shortcut directly.
- **STANDING, paired ABBA across the two binaries, N=15:** 7/15, z = -0.26,
  median B/A 1.0187 -> **inside noise = PARITY with ffmpeg**. The single-run
  medians said +8.3% and the minima +1.9% on the same arms; the paired test is
  what settles it, and it says parity. Recorded as parity, NOT as a win.

## D6k - a cross-implementation ratio needs a cross-BINARY paired harness

- **ASKED:** why did the same pair of arms read +8.3% by median and +1.9% by
  minimum?
- **MEASURED:** the reference's own spread was **1.23x** in that run. Whichever
  statistic you preferred decided the verdict.
- **BUILT:** `paired2.ps1` -- ABBA between two DIFFERENT binaries with their own
  arguments, optional constant subtraction for the reference's demux cost,
  reporting the paired win rate and z. `paired.ps1` could only alternate ONE exe
  across two env settings, which cannot express "us vs the reference".
- **LESSON:** Sec.12 says cross-implementation ratios are standing-only, and this
  is why. Two honest statistics disagreed by 6 points on identical data.

## D5e - merging the AC lookups: REFUTED across three varied probes, REVERTED

- **PRIZE:** `decode_fast_ac` misses 80,373 times per 1080p frame (22%), and each
  miss then calls `decode`, which peeks the IDENTICAL bit window again and loads
  a second table. At ~20 cycles a miss that is ~1.6 of ~18.5 Mcycles/frame, so
  **~8.7% overall** on paper. Built on that arithmetic.
- **MEASURED**, paired ABBA vs ffmpeg, N=15 each, checksums identical throughout:

  | probe | change | verdict |
  |---|---|---|
  | 1 | one 8-byte-padded entry carrying both decodings | z **-1.81**, median 0.9727 |
  | 2 | two small tables, ONE peek, refill threshold `<16` | z **-3.36**, median 0.9520 |
  | 3 | two small tables, ONE peek, refill threshold `<8` (original) | z -0.77, median 0.9910 |

- **ANSWER:** the redundant peek was already free -- it is a shift and mask on a
  register the hardware has, and the second table load hits L1. What was NOT free
  were the two things the merge dragged in with it:
  1. padding the combined entry to 8 bytes took the primary table from **512 B to
     2 KB**, and that table is hit on every symbol;
  2. raising the refill threshold to 16 to cover the long-code path made
     `read_bits` fire on the common path instead of only when short of bits.
     Probe 2 -> 3 isolates this: same code, one constant, 0.9520 -> 0.9910.
- **REVERTED because MEASURED WORSE**, not because it sat inside noise (Sec.12) --
  probes 1 and 2 are both decisive, and probe 3 merely returns to where we
  started while adding an enum and a function. Tree restored to the parity state;
  all six fixture checksums and 58/58 tests confirm it.
- **LESSON:** "remove the redundant work" is not automatically a win when the
  redundant work is register-cheap and the removal costs footprint. The prize
  arithmetic was right about the miss COUNT and wrong about the miss PRICE --
  price the operation, not just the frequency.

---

## Standing at the end of this descent

| | vs ffmpeg | how measured |
|---|---|---|
| **encoder** | **1.19x FASTER (16%)** | pinned CPU, matched output size (27.17 vs 28.04 MB), like-for-like fixed Huffman tables, null arm subtracted |
| **decoder** | **PARITY** (13/21, z 1.09, median 1.0278 -> inside noise) | paired ABBA across binaries, N=21, byte-identical bitstream both arms |

Decoder journey this descent: a VOID 1.47x/1.63x (broken arms) -> a valid 1.10x
slower -> **parity**, via the AVX2 pair IDCT (7.8%, the only lever that survived)
and the `set_single_threaded` repair (38% less CPU, and it made `reclaim_buffer`
live). The goal of 10% FASTER is **not met**; entropy is now the dominant stage
and the three cheap levers into it are all refuted on measurements recorded above.

---

## Standing rules for this descent

- Counts come from `--features counters`. Cycles come from `--features profile`.
  **Never both.**
- Wall measurements are pinned (single core, High priority) and prefer CPU time;
  this box drifts ~15% and migrates across cores.
- No lever is refuted on one measurement. Three *varied* probes, and at least
  one at the level above the change.
