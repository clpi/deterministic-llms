# Multi-model, architecture and density report (rebuilt from the log)

Log: 4456 records, chain head 81471bc5437e41b8.

## Six models (tasks/MULTI_CONTRACT.json)

Cells are bitwise / equal integer scores / equal picks.

| model | repeat in process | fresh 1 thread | fresh 4 threads | one call vs per step | cached vs recompute | Lean agree, sparseEq | tamper exit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2-fp32-ort | 192/192/192 of 192 | 192/192/192 of 192 | 192/192/192 of 192 | 840/840/840 of 840 | n/a | 840, 840 of 840 | 1 |
| gpt2-int8-ort | 192/192/192 of 192 | 192/192/192 of 192 | 192/192/192 of 192 | 1/1/633 of 840 | n/a | 840, 840 of 840 | 1 |
| gpt2-numpy-kv | 192/192/192 of 192 | 192/192/192 of 192 | 192/192/192 of 192 | 70/70/840 of 840 | 35/35/840 of 840 | 840, 840 of 840 | 1 |
| t5-base-ort | 44/44/44 of 44 | 44/44/44 of 44 | 44/44/44 of 44 | 171/171/206 of 206 | n/a | 206, 206 of 206 | 1 |
| llama-style-random | 192/192/192 of 192 | 0/0/192 of 192 | 192/192/192 of 192 | 70/70/840 of 840 | 35/35/840 of 840 | 840, 840 of 840 | 1 |
| mamba-style-random | 192/192/192 of 192 | 192/192/192 of 192 | 192/192/192 of 192 | 840/840/840 of 840 | 35/35/840 of 840 | 840, 840 of 840 | 1 |

| model | clean calls | drafting calls | reply mismatches | certificate mismatches | do-nothing calls | GAMED calls | drift vs fp32 picks | equal sequences |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2-fp32-ort | 840 | 577 | 0 | 0 | 840 | 105 | 840/840 | 35/35 |
| gpt2-int8-ort | 840 | 545 | 28 | 35 | 840 | 620 | 377/840 | 0/35 |
| gpt2-numpy-kv | 840 | 577 | 0 | 2 | 840 | 105 | 840/840 | 35/35 |
| t5-base-ort | 206 | 206 | 0 | 0 | 206 | 38 | n/a | n/a |
| llama-style-random | 840 | 838 | 0 | 0 | 840 | 105 | n/a | n/a |
| mamba-style-random | 840 | 840 | 0 | 0 | 840 | 105 | n/a | n/a |

## Architecture as a parameter (tasks/ARCH_CONTRACT.json)

| config | weights | repeat / fresh 1t / fresh 4t | full call vs step | incremental vs step | Lean agree, sparseEq | tamper | drafting calls (clean 840), reply / cert mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 attn-learned-mha-gelu-ln | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/548/840 of 840 | 0/452/840 of 840 | 840, 840 of 840 | 1 | 464, 0 / 0 |
| A1 attn-learned-mha-gelu-ln | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/401/840 of 840 | 0/274/840 of 840 | 840, 840 of 840 | 1 | 788, 0 / 1 |
| A2 attn-rope-gqa-swiglu-rms | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/463/840 of 840 | 0/335/840 of 840 | 840, 840 of 840 | 1 | 462, 0 / 0 |
| A2 attn-rope-gqa-swiglu-rms | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/281/840 of 840 | 0/165/840 of 840 | 840, 840 of 840 | 1 | 824, 0 / 0 |
| A3 attn-rope-mqa-swiglu-rms | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/508/840 of 840 | 0/368/840 of 840 | 840, 840 of 840 | 1 | 464, 0 / 0 |
| A3 attn-rope-mqa-swiglu-rms | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/281/840 of 840 | 0/152/840 of 840 | 840, 840 of 840 | 1 | 827, 0 / 0 |
| A4 attn-alibi-mha-gelu-ln | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/595/840 of 840 | 0/477/840 of 840 | 840, 840 of 840 | 1 | 465, 0 / 0 |
| A4 attn-alibi-mha-gelu-ln | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/413/840 of 840 | 0/241/840 of 840 | 840, 840 of 840 | 1 | 792, 0 / 0 |
| A5 attn-rope-gqa-moe-rms | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 70/502/840 of 840 | 0/365/840 of 840 | 840, 840 of 840 | 1 | 462, 0 / 0 |
| A5 attn-rope-gqa-moe-rms | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 71/372/840 of 840 | 0/225/840 of 840 | 840, 840 of 840 | 1 | 810, 0 / 0 |
| S1 ssm-selective | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/357/840 of 840 | 840, 840 of 840 | 1 | 456, 0 / 0 |
| S1 ssm-selective | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/137/840 of 840 | 840, 840 of 840 | 1 | 825, 0 / 0 |
| R1 rnn-rwkv | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/405/840 of 840 | 840, 840 of 840 | 1 | 476, 0 / 0 |
| R1 rnn-rwkv | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/227/840 of 840 | 840, 840 of 840 | 1 | 815, 0 / 0 |
| R2 rnn-gru | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/473/840 of 840 | 840, 840 of 840 | 1 | 574, 0 / 0 |
| R2 rnn-gru | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/305/840 of 840 | 840, 840 of 840 | 1 | 823, 0 / 0 |
| R3 rnn-lstm | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/496/840 of 840 | 840, 840 of 840 | 1 | 495, 0 / 0 |
| R3 rnn-lstm | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 840/840/840 of 840 | 0/287/840 of 840 | 840, 840 of 840 | 1 | 814, 0 / 0 |
| H1 hybrid-attn-ssm | random | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 73/427/840 of 840 | 0/296/840 of 840 | 840, 840 of 840 | 1 | 472, 0 / 0 |
| H1 hybrid-attn-ssm | trained | 192/192/192 of 192; 192/192/192 of 192; 192/192/192 of 192 | 72/305/840 of 840 | 0/154/840 of 840 | 840, 840 of 840 | 1 | 816, 0 / 0 |

Trained tiny models (600 Adam steps on the 15,685-token TRAIN corpus), T1 with report.py's metrics:

| config | params | final train loss | ref_match /24 | ref_f1 | rep3 | distinct2 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 attn-learned-mha-gelu-ln | 1231104 | 0.1713 | 0.20 | 0.104 | 0.039 | 0.934 |
| A2 attn-rope-gqa-swiglu-rms | 1101568 | 0.1919 | 0.34 | 0.101 | 0.005 | 0.983 |
| A3 attn-rope-mqa-swiglu-rms | 1068800 | 0.1886 | 0.40 | 0.124 | 0.000 | 0.993 |
| A4 attn-alibi-mha-gelu-ln | 1165568 | 0.1503 | 0.14 | 0.114 | 0.039 | 0.929 |
| A5 attn-rope-gqa-moe-rms | 2688768 | 0.1655 | 0.31 | 0.112 | 0.021 | 0.964 |
| S1 ssm-selective | 1086208 | 0.1514 | 0.17 | 0.098 | 0.005 | 0.981 |
| R1 rnn-rwkv | 1232128 | 0.0915 | 0.14 | 0.113 | 0.013 | 0.974 |
| R2 rnn-gru | 1298688 | 0.0839 | 0.34 | 0.127 | 0.006 | 0.973 |
| R3 rnn-lstm | 1429760 | 0.0892 | 0.20 | 0.110 | 0.014 | 0.969 |
| H1 hybrid-attn-ssm | 1358592 | 0.1216 | 0.49 | 0.104 | 0.023 | 0.970 |

## Density as the computed consequence of the weight change (tasks/DENSITY_CONTRACT.json)

349920 step decisions (243 weight changes x 1440 steps): sparse rule vs dense recompute disagreed on 0.
Fronts recomputed after each of 243 appends; every new front was within the previous front plus the new point: True; Lean exit 0 every time: True.

Final front (minimize work, ban hits, topic misses on held-out words, end misses, loops, drift):

| point (weights in logits) | density | work/step | ban hits | topic misses | end misses | loops | drift | Chebyshev weights aimed here pick | Lean agree, sparseEq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.00000 | 1.0 | 91 | 480 | 438 | 465 | 0 | d_rep0_noloop0_ban0_topic0_end0 | 1440, 1440 of 1440 |
| end2 | 0.00118 | 60.36 | 91 | 480 | 407 | 461 | 32 | d_rep0_noloop0_ban0_topic0_end2 | 1440, 1440 of 1440 |
| end8 | 0.00118 | 60.36 | 91 | 480 | 102 | 383 | 1737 | d_rep0_noloop0_ban0_topic0_end8 | 1440, 1440 of 1440 |
| end8, topic8 | 0.00126 | 64.36 | 91 | 480 | 102 | 380 | 1973 | d_rep0_noloop0_ban0_topic8_end8 | 1440, 1440 of 1440 |
| ban2 | 0.00030 | 16.43 | 34 | 480 | 438 | 458 | 51 | d_rep0_noloop0_ban2_topic0_end0 | 1440, 1440 of 1440 |
| ban2, end2 | 0.00149 | 75.79 | 34 | 480 | 407 | 454 | 84 | d_rep0_noloop0_ban2_topic0_end2 | 1440, 1440 of 1440 |
| ban2, end8 | 0.00149 | 75.79 | 34 | 480 | 102 | 376 | 1789 | d_rep0_noloop0_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8 | 0.00030 | 16.43 | 3 | 480 | 438 | 444 | 162 | d_rep0_noloop0_ban8_topic0_end0 | 1440, 1440 of 1440 |
| ban8, end2 | 0.00149 | 75.79 | 3 | 480 | 407 | 440 | 195 | d_rep0_noloop0_ban8_topic0_end2 | 1440, 1440 of 1440 |
| ban8, end8 | 0.00149 | 75.79 | 3 | 480 | 102 | 362 | 1899 | d_rep0_noloop0_ban8_topic0_end8 | 1440, 1440 of 1440 |
| ban8, end8, topic8 | 0.00156 | 79.79 | 3 | 480 | 102 | 359 | 2135 | d_rep0_noloop0_ban8_topic8_end8 | 1440, 1440 of 1440 |
| noloop2 | 0.00001 | 1.75 | 88 | 480 | 438 | 327 | 154 | d_rep0_noloop2_ban0_topic0_end0 | 1440, 1440 of 1440 |
| end2, noloop2 | 0.00119 | 61.06 | 88 | 480 | 405 | 325 | 192 | d_rep0_noloop2_ban0_topic0_end2 | 1440, 1440 of 1440 |
| end8, noloop2 | 0.00119 | 61.1 | 88 | 480 | 78 | 263 | 2060 | d_rep0_noloop2_ban0_topic0_end8 | 1440, 1440 of 1440 |
| noloop2, topic8 | 0.00009 | 5.75 | 88 | 480 | 438 | 320 | 458 | d_rep0_noloop2_ban0_topic8_end0 | 1440, 1440 of 1440 |
| end2, noloop2, topic8 | 0.00127 | 65.06 | 88 | 480 | 405 | 318 | 496 | d_rep0_noloop2_ban0_topic8_end2 | 1440, 1440 of 1440 |
| end8, noloop2, topic8 | 0.00127 | 65.1 | 88 | 480 | 78 | 256 | 2364 | d_rep0_noloop2_ban0_topic8_end8 | 1440, 1440 of 1440 |
| ban2, noloop2 | 0.00031 | 17.16 | 26 | 480 | 438 | 323 | 216 | d_rep0_noloop2_ban2_topic0_end0 | 1440, 1440 of 1440 |
| ban2, end2, noloop2 | 0.00149 | 76.47 | 26 | 480 | 405 | 321 | 254 | d_rep0_noloop2_ban2_topic0_end2 | 1440, 1440 of 1440 |
| ban2, end8, noloop2 | 0.00149 | 76.5 | 26 | 480 | 78 | 259 | 2121 | d_rep0_noloop2_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8, noloop2 | 0.00031 | 17.16 | 1 | 480 | 438 | 315 | 321 | d_rep0_noloop2_ban8_topic0_end0 | 1440, 1440 of 1440 |
| ban8, end2, noloop2 | 0.00149 | 76.47 | 1 | 480 | 405 | 313 | 359 | d_rep0_noloop2_ban8_topic0_end2 | 1440, 1440 of 1440 |
| ban8, end8, noloop2 | 0.00149 | 76.5 | 1 | 480 | 78 | 251 | 2227 | d_rep0_noloop2_ban8_topic0_end8 | 1440, 1440 of 1440 |
| ban8, noloop2, topic8 | 0.00039 | 21.16 | 1 | 480 | 438 | 308 | 625 | d_rep0_noloop2_ban8_topic8_end0 | 1440, 1440 of 1440 |
| ban8, end2, noloop2, topic8 | 0.00157 | 80.47 | 1 | 480 | 405 | 306 | 663 | d_rep0_noloop2_ban8_topic8_end2 | 1440, 1440 of 1440 |
| ban8, end8, noloop2, topic8 | 0.00157 | 80.5 | 1 | 480 | 78 | 244 | 2531 | d_rep0_noloop2_ban8_topic8_end8 | 1440, 1440 of 1440 |
| noloop8 | 0.00001 | 1.75 | 86 | 480 | 448 | 27 | 1270 | d_rep0_noloop8_ban0_topic0_end0 | 1440, 1440 of 1440 |
| end2, noloop8 | 0.00119 | 61.1 | 86 | 480 | 405 | 24 | 1338 | d_rep0_noloop8_ban0_topic0_end2 | 1440, 1440 of 1440 |
| end8, noloop8 | 0.00119 | 61.06 | 86 | 480 | 60 | 18 | 3137 | d_rep0_noloop8_ban0_topic0_end8 | 1440, 1440 of 1440 |
| ban2, noloop8 | 0.00031 | 17.16 | 24 | 480 | 448 | 25 | 1338 | d_rep0_noloop8_ban2_topic0_end0 | 1440, 1440 of 1440 |
| ban2, end2, noloop8 | 0.00149 | 76.5 | 24 | 480 | 405 | 22 | 1406 | d_rep0_noloop8_ban2_topic0_end2 | 1440, 1440 of 1440 |
| ban2, end8, noloop8 | 0.00149 | 76.47 | 24 | 480 | 60 | 16 | 3205 | d_rep0_noloop8_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8, noloop8 | 0.00031 | 17.16 | 1 | 480 | 448 | 25 | 1403 | d_rep0_noloop8_ban8_topic0_end0 | 1440, 1440 of 1440 |
| ban8, end2, noloop8 | 0.00149 | 76.5 | 1 | 480 | 405 | 22 | 1472 | d_rep0_noloop8_ban8_topic0_end2 | 1440, 1440 of 1440 |
| ban8, end8, noloop8 | 0.00149 | 76.47 | 1 | 480 | 60 | 16 | 3271 | d_rep0_noloop8_ban8_topic0_end8 | 1440, 1440 of 1440 |
| rep2 | 0.00063 | 33.53 | 77 | 480 | 467 | 216 | 1136 | d_rep2_noloop0_ban0_topic0_end0 | 1440, 1440 of 1440 |
| end2, rep2 | 0.00180 | 92.34 | 77 | 480 | 438 | 218 | 1162 | d_rep2_noloop0_ban0_topic0_end2 | 1440, 1440 of 1440 |
| end8, rep2 | 0.00180 | 92.38 | 77 | 480 | 96 | 170 | 3026 | d_rep2_noloop0_ban0_topic0_end8 | 1440, 1440 of 1440 |
| ban2, rep2 | 0.00090 | 47.57 | 20 | 480 | 467 | 211 | 1195 | d_rep2_noloop0_ban2_topic0_end0 | 1440, 1440 of 1440 |
| ban2, end2, rep2 | 0.00207 | 106.38 | 20 | 480 | 438 | 213 | 1221 | d_rep2_noloop0_ban2_topic0_end2 | 1440, 1440 of 1440 |
| ban2, end8, rep2 | 0.00207 | 106.42 | 20 | 480 | 96 | 165 | 3085 | d_rep2_noloop0_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8, rep2 | 0.00090 | 47.57 | 1 | 480 | 467 | 206 | 1279 | d_rep2_noloop0_ban8_topic0_end0 | 1440, 1440 of 1440 |
| ban8, end2, rep2 | 0.00207 | 106.38 | 1 | 480 | 438 | 208 | 1305 | d_rep2_noloop0_ban8_topic0_end2 | 1440, 1440 of 1440 |
| ban8, end8, rep2 | 0.00207 | 106.42 | 1 | 480 | 96 | 160 | 3169 | d_rep2_noloop0_ban8_topic0_end8 | 1440, 1440 of 1440 |
| noloop8, rep2 | 0.00063 | 33.53 | 77 | 480 | 468 | 0 | 2187 | d_rep2_noloop8_ban0_topic0_end0 | 1440, 1440 of 1440 |
| end2, noloop8, rep2 | 0.00180 | 92.34 | 77 | 480 | 434 | 0 | 2229 | d_rep2_noloop8_ban0_topic0_end2 | 1440, 1440 of 1440 |
| end8, noloop8, rep2 | 0.00180 | 92.39 | 77 | 480 | 82 | 0 | 3964 | d_rep2_noloop8_ban0_topic0_end8 | 1440, 1440 of 1440 |
| ban2, noloop8, rep2 | 0.00090 | 47.57 | 16 | 480 | 468 | 0 | 2231 | d_rep2_noloop8_ban2_topic0_end0 | 1440, 1440 of 1440 |
| ban2, end2, noloop8, rep2 | 0.00207 | 106.38 | 16 | 480 | 434 | 0 | 2274 | d_rep2_noloop8_ban2_topic0_end2 | 1440, 1440 of 1440 |
| ban2, end8, noloop8, rep2 | 0.00207 | 106.43 | 16 | 480 | 82 | 0 | 4008 | d_rep2_noloop8_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8, noloop8, rep2 | 0.00090 | 47.57 | 1 | 480 | 468 | 0 | 2282 | d_rep2_noloop8_ban8_topic0_end0 | 1440, 1440 of 1440 |
| ban8, end2, noloop8, rep2 | 0.00207 | 106.38 | 1 | 480 | 434 | 0 | 2324 | d_rep2_noloop8_ban8_topic0_end2 | 1440, 1440 of 1440 |
| ban8, end8, noloop8, rep2 | 0.00207 | 106.43 | 1 | 480 | 82 | 0 | 4059 | d_rep2_noloop8_ban8_topic0_end8 | 1440, 1440 of 1440 |
| end8, rep8 | 0.00180 | 92.34 | 79 | 480 | 66 | 6 | 4615 | d_rep8_noloop0_ban0_topic0_end8 | 1440, 1440 of 1440 |
| ban2, end8, rep8 | 0.00207 | 106.38 | 16 | 480 | 66 | 5 | 4664 | d_rep8_noloop0_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8, end8, rep8 | 0.00207 | 106.38 | 1 | 480 | 66 | 5 | 4713 | d_rep8_noloop0_ban8_topic0_end8 | 1440, 1440 of 1440 |
| end8, noloop2, rep8 | 0.00180 | 92.34 | 79 | 480 | 66 | 0 | 4668 | d_rep8_noloop2_ban0_topic0_end8, d_rep8_noloop8_ban0_topic0_end8 | 1440, 1440 of 1440 |
| ban2, end8, noloop2, rep8 | 0.00207 | 106.38 | 16 | 480 | 66 | 0 | 4708 | d_rep8_noloop2_ban2_topic0_end8, d_rep8_noloop8_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8, end8, noloop2, rep8 | 0.00207 | 106.38 | 1 | 480 | 66 | 0 | 4757 | d_rep8_noloop2_ban8_topic0_end8, d_rep8_noloop8_ban8_topic0_end8 | 1440, 1440 of 1440 |
| end8, noloop8, rep8 | 0.00180 | 92.34 | 79 | 480 | 66 | 0 | 4668 | d_rep8_noloop2_ban0_topic0_end8, d_rep8_noloop8_ban0_topic0_end8 | 1440, 1440 of 1440 |
| ban2, end8, noloop8, rep8 | 0.00207 | 106.38 | 16 | 480 | 66 | 0 | 4708 | d_rep8_noloop2_ban2_topic0_end8, d_rep8_noloop8_ban2_topic0_end8 | 1440, 1440 of 1440 |
| ban8, end8, noloop8, rep8 | 0.00207 | 106.38 | 1 | 480 | 66 | 0 | 4757 | d_rep8_noloop2_ban8_topic0_end8, d_rep8_noloop8_ban8_topic0_end8 | 1440, 1440 of 1440 |
