# April 30: Original benchmark #
> 4b0d78cdf6576ee554204fe4d0124c745ac4f81b

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
> 419c508dba9ff0bc003f9906fdca74dbc87d6380

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

# May 22 2021: GPU using streams even faster, and allows processing of more data #
> b0e451f5e91fa1f587ef08d3469f6b341b3be04c

<details>
<summary>Click this to collapse/fold.</summary>

> `R_POINTS=1000`, `SP_POINTS=400`, `R_POINTS_PER_CHUNK=500`

| Group | Experiment      | Prob. Space | Samples | Iterations | Baseline | us/Iteration | Iterations/sec | RAM (bytes) |
|:-----:|:---------------:|:-----------:|:-------:|:----------:|:--------:|:------------:|:--------------:|:-----------:|
| POWER | 1T_NO_BACK      | Null        | 100     | 100        | 1.00000  | 9366.87000   | 106.76         | 9426067456  |
| POWER | 2T_NO_BACK      | Null        | 100     | 100        | 1.17076  | 10966.31000  | 91.19          | 9434460160  |
| POWER | 8T_NO_BACK      | Null        | 100     | 100        | 1.20283  | 11266.78000  | 88.76          | 9451245568  |
| POWER | 1T_NO_BACK_FULL | Null        | 100     | 100        | 1.51296  | 14171.72000  | 70.56          | 9451245568  |
| POWER | 1T_CONST_BACK   | Null        | 100     | 100        | 1.00754  | 9437.45000   | 105.96         | 9451245568  |
| POWER | 1T_CONST_BACK_F | Null        | 100     | 100        | 1.47449  | 13811.38000  | 72.40          | 9451245568  |
| POWER | 1T_BACK         | Null        | 100     | 100        | 1.25850  | 11788.16000  | 84.83          | 9452847104  |
| POWER | 1T_BACK_FULL_MA | Null        | 100     | 100        | 1.70077  | 15930.87000  | 62.77          | 9453252608  |
| POWER | GPU_V1          | Null        | 100     | 100        | 0.04450  | 416.86000    | 2398.89        | 9453658112  |
| POWER | GPU_V2          | Null        | 100     | 100        | 0.03995  | 374.20000    | 2672.37        | 9453658112  |

</details>

# May 25 2021: Comparing Digitiser read speed to GPU #
> 68079aea94f892bbac75a7de04f7b434824ffee0

Theoretically the digitise will work at:
- 400 (1000ns) points, the repetition frequency is 340kHz, so for 254200 records this is `0.74s`
- 200 (500ns) points, the repetition frequency is 860kHz, so for 254200 records this is `0.3s`

It seems that the GPU kernel peaks at 2 streams in terms of performance.

<details>
<summary>Click this to collapse/fold.</summary>

> `R_POINTS=128000`, `SP_POINTS=400`, `R_POINTS_PER_CHUNK=1000`

| Group     | Experiment      | Prob. Space | Samples | Iterations | Baseline | us/Iteration  | Iterations/sec | RAM (bytes) |
|:---------:|:---------------:|:-----------:|:-------:|:----------:|:--------:|:-------------:|:--------------:|:-----------:|
| DIGITISER | Theoretical     |             |         |            |          | 740,000       | 1.35           |             |
| POWER     | 1T_NO_BACK      | Null        | 30      | 1          | 1.00000  | 1927193.00000 | 0.52           | 60559360    |
| POWER     | 2T_NO_BACK      | Null        | 30      | 1          | 1.31233  | 2529108.00000 | 0.40           | 68952064    |
| POWER     | 8T_NO_BACK      | Null        | 30      | 1          | 1.40540  | 2708469.00000 | 0.37           | 85737472    |
| POWER     | 1T_NO_BACK_FULL | Null        | 30      | 1          | 1.69837  | 3273092.00000 | 0.31           | 85737472    |
| POWER     | 1T_CONST_BACK   | Null        | 30      | 1          | 1.05101  | 2025497.00000 | 0.49           | 85737472    |
| POWER     | 1T_CONST_BACK_F | Null        | 30      | 1          | 1.75032  | 3373199.00000 | 0.30           | 85737472    |
| POWER     | 1T_BACK         | Null        | 30      | 1          | 1.32643  | 2556294.00000 | 0.39           | 85737472    |
| POWER     | 1T_BACK_FULL_MA | Null        | 30      | 1          | 2.01473  | 3882780.00000 | 0.26           | 85737472    |
| POWER     | GPU_1ST         | Null        | 30      | 20         | 0.02414  | 46523.55000   | 21.49          | 9418448896  |
| POWER     | GPU_2ST         | Null        | 30      | 29         | 0.01728  | 33303.72414   | 30.03          | 9418448896  |
| POWER     | GPU_8ST         | Null        | 30      | 30         | 0.01706  | 32872.36667   | 30.42          | 9418448896  |
| POWER     | GPU_16ST        | Null        | 30      | 30         | 0.01719  | 33132.93333   | 30.18          | 9552666624  |

</details>


# Float vs double: No difference #
> 4b2b92bdcc7e9de0fce5234998fde309ac71d137

<details>
<summary>Results</summary>

| Group          | Experiment      | Prob. Space | Samples | Iterations | Baseline | us/Iteration | Iterations/sec | RAM (bytes) |
|:--------------:|:---------------:|:-----------:|:-------:|:----------:|:--------:|:------------:|:--------------:|:-----------:|
| TYPE_BENCHMARK | FLOAT_MULITPLY  | Null        | 1000    | 50000      | 1.00000  | 2.64130      | 378601.45      | 51908608    |
| TYPE_BENCHMARK | DOUBLE_MULTIPLY | Null        | 1000    | 50000      | 1.00096  | 2.64384      | 378237.71      | 51908608    |
| TYPE_BENCHMARK | FLOAT_ADD       | Null        | 1000    | 50000      | 0.91550  | 2.41812      | 413544.41      | 51908608    |
| TYPE_BENCHMARK | DOUBLE_ADD      | Null        | 1000    | 50000      | 0.91822  | 2.42530      | 412320.13      | 51908608    |

</details>
