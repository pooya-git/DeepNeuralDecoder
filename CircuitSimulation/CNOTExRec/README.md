Steane EC training set data interpretation:

1) File names include the physical failure rate p. For example, SyndromeAndError1.000e-03.txt contains the syndromes and errors of the CNOT exRec circuit where the noise model is given by a depolarizing channel (see page 3 action B of  https://arxiv.org/abs/1708.02246 for a full description of the circuit level depolarizing noise channel) and p = 10^-3.

2) Each line contains a bit string with the following format (SynX,SynZ,ErrX,ErrZ) where SynX is the X error syndrome, SynZ is the Z error syndrome, ErrX is the X error on the data and ErrZ is the Z error on the data. Note that for the [[7,1,3]] code, SynX and SynZ are bit strings of length 3 where as ErrX and ErrZ are bit strings of length 7 each. So a row has 20 bits.

3) Since a CNOT exRec has 4 blocks, for a given iteration the data is divided into 4 lines. The first line has (SynX,SynZ,ErrX,ErrZ) for the errors on the control qubit prior to the logical CNOT (before the second EC), the second line contains (SynX,SynZ,ErrX,ErrZ) for the errors on the target qubit prior to the logical CNOT (before the second EC). The last two lines contain (SynX,SynZ,ErrX,ErrZ) for the output control and target errors of the CNOT exRec.

4) From 3), for 10^7 iterations, there will be 4*10^7 lines in a given .txt file (4 lines for the two control and target qubit locations of the CNOT exRec).

Surface code training set data:

1) In this case I separated the X and Z syndromes as well as the X and Z error outcomes into 4 separate .txt file (for a given p of the depolarizing noise channel).

2) The X error syndrome is given as a matrix of size ((d+1)/2,d-1) and the Z error syndrome is given of size (d-1,(d+1)/2).

3) For a given d, and a particular iteration, there will be d syndrome X and Z matrices which shows how the syndrome changes on all d rounds. For say m iterations (taken to be 10^6 in this case), the XSyndromep.txt files will have d-1 columns and m*d*(d+1)/2 rows. The  ZSyndromep.txt files will have (d+1)/2 columns and m*d*(-1) rows.

4) The files Xerrorp.txt and Zerrorp.txt contain the associated errors (in groups of d) corresponding to the X and Z data errors on a given round. For a distance d surface code, it has d^2 physical qubits, so each row has d^2 bits. The for m iterations, the file will contain d*m rows.