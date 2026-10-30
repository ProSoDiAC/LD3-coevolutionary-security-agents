# 2x2 language-model campaign (attacker memory x defender memory)

## recurring
| cell | n | ASR | DR | ER | DR|E | gate |
|---|--:|--:|--:|--:|--:|--:|
| A0D0 | 20 | 0.398 | 0.470 | 0.973 | 0.483 | 0.714 |
| A0D1 | 20 | 0.343 | 0.517 | 0.973 | 0.531 | 0.714 |
| A1D0 | 20 | 0.230 | 0.492 | 0.990 | 0.497 | 0.592 |
| A1D1 | 20 | 0.147 | 0.557 | 0.970 | 0.575 | 0.625 |

### effect on ASR (recurring)
- attacker_memory: Delta=-0.182 (n=20, p=<.001, p_BH=<.001, 95% CI [-0.214, -0.151])
- defender_memory: Delta=-0.069 (n=20, p=<.001, p_BH=0.002, 95% CI [-0.103, -0.037])
- interaction: Delta=-0.028 (n=20, p=0.350, p_BH=0.374, 95% CI [-0.082, +0.032])

### effect on DR (recurring)
- attacker_memory: Delta=+0.031 (n=20, p=0.125, p_BH=0.187, 95% CI [-0.007, +0.066])
- defender_memory: Delta=+0.056 (n=20, p=0.001, p_BH=0.002, 95% CI [+0.028, +0.084])
- interaction: Delta=+0.018 (n=20, p=0.374, p_BH=0.374, 95% CI [-0.020, +0.058])

## iid
| cell | n | ASR | DR | ER | DR|E | gate |
|---|--:|--:|--:|--:|--:|--:|
| A0D0 | 20 | 0.417 | 0.513 | 0.988 | 0.520 | 0.708 |
| A0D1 | 20 | 0.357 | 0.565 | 0.988 | 0.572 | 0.708 |
| A1D0 | 20 | 0.243 | 0.480 | 0.993 | 0.483 | 0.560 |
| A1D1 | 20 | 0.182 | 0.595 | 0.983 | 0.605 | 0.614 |

### effect on ASR (iid)
- attacker_memory: Delta=-0.174 (n=20, p=<.001, p_BH=<.001, 95% CI [-0.210, -0.138])
- defender_memory: Delta=-0.061 (n=20, p=<.001, p_BH=<.001, 95% CI [-0.081, -0.041])
- interaction: Delta=-0.002 (n=20, p=0.952, p_BH=0.952, 95% CI [-0.052, +0.052])

### effect on DR (iid)
- attacker_memory: Delta=-0.002 (n=20, p=0.909, p_BH=0.952, 95% CI [-0.030, +0.025])
- defender_memory: Delta=+0.083 (n=20, p=<.001, p_BH=<.001, 95% CI [+0.062, +0.105])
- interaction: Delta=+0.063 (n=20, p=0.014, p_BH=0.021, 95% CI [+0.020, +0.110])

