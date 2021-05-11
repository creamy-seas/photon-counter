# April 30: Original benchmark #
> cf435b68349770b811ffaa2214983431e1557554

<details>
<summary>Results</summary>

| Group | Experiment      | Prob. Space | Samples | Iterations | Baseline | us/Iteration | Iterations/sec | RAM (bytes) |
|:-----:|:---------------:|:-----------:|:-------:|:----------:|:--------:|:------------:|:--------------:|:-----------:|
| POWER | CPU_1T_NO_BACK  | Null        | 10      | 1000       | 1.00000  | 993.03200    | 1007.02        | 60456960    |
| POWER | CPU_2T_NO_BACK  | Null        | 10      | 1000       | 1.15689  | 1148.83300   | 870.45         | 68849664    |
| POWER | CPU_4T_NO_BACK  | Null        | 10      | 1000       | 1.30622  | 1297.12200   | 770.94         | 85635072    |
| POWER | CPU_8T_NO_BACK  | Null        | 10      | 1000       | 1.23859  | 1229.96300   | 813.03         | 86437888    |
| POWER | GPU_NO_BACK     | Null        | 10      | 1000       | 0.26119  | 259.36900    | 3855.51        | 9418870784  |
| POWER | CPU_1T_CONST_BA | Null        | 10      | 1000       | 1.04712  | 1039.82200   | 961.70         | 10216804352 |
| POWER | GPU_CONST_BACK  | Null        | 10      | 1000       | 0.26043  | 258.61800    | 3866.71        | 13816946688 |
| POWER | CPU_1T_BACK     | Null        | 10      | 1000       | 1.81312  | 1800.49100   | 555.40         | 14617780224 |
| POWER | GPU_BACK        | Null        | 10      | 1000       | 0.25992  | 258.11300    | 3874.27        | 18287763456 |

</details>


# May 11 2021: GPU better even when processing everything#
> cbfe20c1d984c7d8a527f57cfd6dc981290dbe0e

<details>
<summary>Results</summary>

|     Group      |   Experiment    |   Prob. Space   |     Samples     |   Iterations    |    Baseline     |  us/Iteration   | Iterations/sec  |   RAM (bytes)   |
|:--------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|POWER           | CPU_1T_NO_BACK  |            Null |              10 |            1000 |         1.00000 |      3194.35900 |          313.05 |      9418752000 |
|POWER           | CPU_2T_NO_BACK  |            Null |              10 |            1000 |         1.03164 |      3295.41400 |          303.45 |      9427144704 |
|POWER           | CPU_8T_NO_BACK  |            Null |              10 |            1000 |         0.94265 |      3011.16900 |          332.10 |      9443930112 |
|POWER           | CPU_1T_NO_BACK_ |            Null |              10 |            1000 |         1.04948 |      3352.40700 |          298.29 |      9443930112 |
|POWER           | CPU_1T_CONST_BA |            Null |              10 |            1000 |         1.27478 |      4072.10900 |          245.57 |      9443930112 |
|POWER           | CPU_1T_CONST_BA |            Null |              10 |            1000 |         1.35867 |      4340.09500 |          230.41 |      9443930112 |
|POWER           | CPU_1T_BACK     |            Null |              10 |            1000 |         0.99066 |      3164.51900 |          316.00 |      9444331520 |
|POWER           | CPU_1T_BACK_FUL |            Null |              10 |            1000 |         1.28411 |      4101.90000 |          243.79 |      9444872192 |
|POWER           | GPU_BACK        |            Null |              10 |            1000 |         0.10027 |       320.30400 |         3122.03 |      9445277696 |

</details>

# Float vs double: No difference #
> 4b2b92bdcc7e9de0fce5234998fde309ac71d137

<details>
<summary>Results</summary>

| Group          | Experiment      | Prob. Space | Samples | Iterations | Baseline | us/Iteration | Iterations/sec | RAM (bytes) |
|:--------------:|:---------------:|:-----------:|:-------:|:----------:|:--------:|:------------:|:--------------:|:-----------:|
| TYPE_BENCHMARK | FLOAT_MULITPLY  | Null        | 1000    | 50000      | 1.00000  | 2.64130      | 378601.45      | 51908608    |
| TYPE_BENCHMARK | FLOAT_ADD       | Null        | 1000    | 50000      | 0.91550  | 2.41812      | 413544.41      | 51908608    |
| TYPE_BENCHMARK | DOUBLE_ADD      | Null        | 1000    | 50000      | 0.91822  | 2.42530      | 412320.13      | 51908608    |
| TYPE_BENCHMARK | DOUBLE_MULTIPLY | Null        | 1000    | 50000      | 1.00096  | 2.64384      | 378237.71      | 51908608    |

</details>
