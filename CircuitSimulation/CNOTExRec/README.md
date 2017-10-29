# Steane EC training dataset:

1) File names include the physical failure rate p. For example,
   SyndromeAndError1.000e-03.txt contains the syndromes and errors of the CNOT
   exRec circuit where the noise model is given by a depolarizing channel (see
   page 3 action B of  https://arxiv.org/abs/1708.02246 for a full description
   of the circuit level depolarizing noise channel) and p = 10^-3.

2) Each line contains a bit string with the following format
   (SynX,SynZ,ErrX,ErrZ) where SynX is the X error syndrome, SynZ is the Z error
   syndrome, ErrX is the X error on the data and ErrZ is the Z error on the
   data. Note that for the [[7,1,3]] code, SynX and SynZ are bit strings of
   length 3 where as ErrX and ErrZ are bit strings of length 7 each. So a row
   has 20 bits.

3) Since a CNOT exRec has 4 blocks, for a given iteration the data is divided
   into 4 lines. The first line has (SynX,SynZ,ErrX,ErrZ) for the errors on the
   control qubit prior to the logical CNOT (before the second EC), the second
   line contains (SynX,SynZ,ErrX,ErrZ) for the errors on the target qubit prior
   to the logical CNOT (before the second EC). The last two lines contain
   (SynX,SynZ,ErrX,ErrZ) for the output control and target errors of the CNOT
   exRec.

4) From 3), for 10^7 iterations, there will be 4*10^7 lines in a given .txt file
   (4 lines for the two control and target qubit locations of the CNOT exRec).
