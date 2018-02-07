# ML-based Fault Tolerant Decoders
Goal: To perform QECC decoding at the circuit level.

Hypertuning with categorical variables (activations):

| Scheme  |  Type  | D | RNN | FF0 | FF1 | FF2 | FF3 | CNN |
| ------- | ------ | - | --- | --- | --- | --- | --- | --- |
| Steane  | PureEr | 3 | Yes | Yes | Yes | Yes | Yes |     |
| Steane  | Lookup | 3 | Yes | Yes | Yes | Yes | Yes |     |
| Knill   | PureEr | 3 |     |     |     |     |     |     |
| Knill   | LookUp | 3 |     |     |     |     |     |     |
| Surface | PureEr | 3 |     |     |     |     |     |     |
| Surface | LookUp | 3 |     |     |     |     |     |     |
| Steane  | PureEr | 5 |     |     |     |     |     |     |
| Steane  | Lookup | 5 |     |     |     |     |     |     |
| Knill   | PureEr | 5 |     |     |     |     |     |     |
| Knill   | LookUp | 5 |     |     |     |     |     |     |
| Surface | PureEr | 5 |     |     |     |     |     |     |
| Surface | LookUp | 5 |     |     |     |     |     |     |

Conclusion: Bayesopt cannot combine categorical and continuous variables. 

Hypertuning with grid search on activations:

| Scheme  |  Type  | D | RNN | FF0 | FF1 | FF2 | FF3 | CNN |
| ------- | ------ | - | --- | --- | --- | --- | --- | --- |
| Steane  | PureEr | 3 |     | Yes | Yes | Yes |     |     |
| Steane  | Lookup | 3 |     | Yes | Yes | Yes |     |     |
| Knill   | PureEr | 3 |     |     |     |     |     |     |
| Knill   | LookUp | 3 |     |     |     |     |     |     |
| Surface | PureEr | 3 |     |     |     |     |     |     |
| Surface | LookUp | 3 |     |     |     |     |     |     |
| Steane  | PureEr | 5 |     |     |     |     |     |     |
| Steane  | Lookup | 5 |     |     |     |     |     |     |
| Knill   | PureEr | 5 |     |     |     |     |     |     |
| Knill   | LookUp | 5 |     |     |     |     |     |     |
| Surface | PureEr | 5 |     |     |     |     |     |     |
| Surface | LookUp | 5 |     |     |     |     |     |     |

Conclusion: The relu activation function is a good choice.

Hypertuning with relu activations:

| Scheme  |  Type  | D | RNN | FF0 | FF1 | FF2 | FF3 | CNN |  B Range  | Tune |
| ------- | ------ | - | --- | --- | --- | --- | --- | --- | --------- | ---- |
| Steane  | PureEr | 3 | Yes | Yes | Yes | Yes |     |     | 1e-4 6e-4 | 4e-4 |
| Steane  | Lookup | 3 | Yes | Yes | Yes | Yes |     |     | 1e-4 6e-4 | 4e-4 |
| Knill   | PureEr | 3 |     |     | Yes |     |     |     | 1e-4 6e-4 | 4e-4 |
| Knill   | LookUp | 3 |     |     | Yes |     |     |     | 1e-4 6e-4 | 4e-4 |
| Surface | PureEr | 3 | Yes | Yes | Yes | Yes |     |     | 1e-4 6e-4 | 4e-4 |
| Surface | LookUp | 3 | Yes | Yes | Yes | Yes |     |     | 1e-4 6e-4 | 4e-4 |
| SurBias | PureEr | 3 |     |     | Yes |     |     |     | 1e-4 6e-4 | 4e-4 |
| SurBias | LookUp | 3 |     |     | Yes |     |     |     | 1e-4 6e-4 | 4e-4 |
| Steane  | PureEr | 5 | Yes |     | Yes | Yes |     |     | 6e-4 2e-3 | 4e-4 |
| Steane  | Lookup | 5 | Yes |     | Yes | Yes |     |     | 6e-4 2e-3 | 4e-4 |
| Knill   | PureEr | 5 |     |     | Yes |     |     |     | 6e-4 2e-3 | 4e-4 |
| Knill   | LookUp | 5 |     |     | Yes |     |     |     | 6e-4 2e-3 | 4e-4 |
| Surface | PureEr | 5 |     |     | Yes |     | Yes | Yes | 3e-4 8e-4 | 4e-4 |
| Surface | LookUp | 5 |     |     | Yes |     | Yes | Yes | 3e-4 8e-4 | 4e-4 |

After Chris' debugging:

| Scheme | T  | D | RNN | FF0 | FF1 | FF2 | FF3 | FF4 | FF5 | CNN |  B Range  | Tune |
| ------ | -- | - | --- | --- | --- | --- | --- | --- | --- | --- | --------- | ---- |
| Steane | PU | 3 | Yes |     | Yes | Yes | Yes |  -  |     |     | 1e-4 6e-4 | 4e-4 |
| Steane | LU | 3 | Yes | Yes | Yes | Yes | Yes | Yes |  -  |     | 1e-4 6e-4 | 4e-4 |
| Knill  | PU | 3 | Yes |     |     | Yes | Yes |  -  |     |     | 1e-4 6e-4 | 4e-4 |
| Knill  | LU | 3 | Yes | Yes | Yes | Yes | Yes | Yes*|  -  |     | 1e-4 6e-4 | 4e-4 |
| Sur    | PU | 3 | Yes | Yes | Yes | Yes |  -  |     |     |     | 1e-4 6e-4 | 4e-4 |
| Sur    | LU | 3 | Yes | Yes | Yes | Yes |  -  |     |     |     | 1e-4 6e-4 | 4e-4 |
| Steane | PU | 5 | Yes |     |     | Yes | Yes |  -  |     |     | 6e-4 2e-3 | 1e-3 |
| Steane | LU | 5 | Yes |     |     | Yes | Yes*|  -  |     |     | 6e-4 2e-3 | 1e-3 |
| Knill  | PU | 5 |     |     |     | Yes |     |     |     |     | 6e-4 2e-3 | 4e-4 |
| Knill  | LU | 5 |     |     |     | Yes |     |     |     |     | 6e-4 2e-3 | 4e-4 |
| Sur    | PU | 5 | K3T |     |     | Yes |     |     |     | K2T | 3e-4 8e-4 | 6e-4 |
| Sur    | LU | 5 | G4T |     |     | Yes |     |     |     | G3T | 3e-4 8e-4 | 6e-4 |
| SurB05 | PU | 3 |     |     | Yes |     |     |     |     |     | 1e-4 6e-4 | 4e-4 |
| SurB05 | LU | 3 |     |     | Yes |     |     |     |     |     | 1e-4 6e-4 | 4e-4 |
| SurB10 | PU | 3 |     |     | Yes |     |     |     |     |     | 1e-4 6e-4 | 4e-4 |
| SurB10 | LU | 3 |     |     | Yes |     |     |     |     |     | 1e-4 6e-4 | 4e-4 |

* no json due to runtime interruption. 

