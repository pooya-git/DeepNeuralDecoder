# Surface code training dataset:

1) In this case I separated the X and Z syndromes as well as the X and Z error
   outcomes into 4 separate .txt files (for a given p of the depolarizing noise
   channel).

2) The X error syndrome is given as a matrix of size ((d+1)/2,d-1) and the Z
   error syndrome is given of size (d-1,(d+1)/2).

3) For a given d, and a particular iteration, there will be d syndrome X and Z
   matrices which shows how the syndrome changes on all d rounds. For say m
   iterations (taken to be 10^6 in this case), the XSyndromep.txt files will
   have d-1 columns and m*d*(d+1)/2 rows. The  ZSyndromep.txt files will have
   (d+1)/2 columns and m*d*(-1) rows.

4) The files Xerrorp.txt and Zerrorp.txt contain the associated errors (in
   groups of d) corresponding to the X and Z data errors on a given round. For a
   distance d surface code, it has d^2 physical qubits, so each row has d^2
   bits. The for m iterations, the file will contain d*m rows.