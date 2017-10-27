function Test


% Here err represents the sum (mod 2) of the error from the training set
% and the correction output from the machine learning decoder.
err = zeros(1,7);
err(1,7) = 1;


% The goal of the function CheckLogicalSteane(err) is to verify if err is
% a correctable error or not.
CheckLogicalSteane(err)

% The function
% ComputeLogicalUsingLookupTableX(SynX1,SynX2,SynX3,SynX4,Err3X,Err4X)
% outputs 1 if either there is a logical X fault on the third or fourth block
% after performing a correction using the lookup table.

% The function
% ComputeLogicalUsingLookupTableZ(SynZ1,SynZ2,SynZ3,SynZ4,Err3Z,Err4Z)
% outputs 1 if either there is a logical Z fault on the third or fourth block
% after performing a correction using the lookup table.
end

function Output = CheckLogicalSteane(err)

% This function outputs 0 if err is a correctable error and 1 otherwise.

g = [0,0,0,1,1,1,1
    0,1,1,0,0,1,1
    1,0,1,0,1,0,1];

MatRecovery = [0,0,0,0,0,0,0
    1,0,0,0,0,0,0
    0,1,0,0,0,0,0
    0,0,1,0,0,0,0
    0,0,0,1,0,0,0
    0,0,0,0,1,0,0
    0,0,0,0,0,1,0
    0,0,0,0,0,0,1];

% Compute the syndrome of err
syn = zeros(1,3);
for i = 1:length(g(:,1))
    syn(1,i) = mod(sum(conj(err).* g(i,:)),2);     
end

% Find the correction corresponding to the syndrome
correctionRow = 1;
for i = 1:length(syn)
    correctionRow = correctionRow + 2^(3-i)*syn(i);
end

corr = MatRecovery(correctionRow,:);

% Outputs 0 if err + corr is a stabilizer and 1 if it is a logical fault
Output = mod(sum(mod(err+corr,2)),2);

end

function Output = equivalenceTable(err) % used for measurments in the Z basis
g = [0,0,0,1,1,1,1
    0,1,1,0,0,1,1
    1,0,1,0,1,0,1];

MatRecovery = [0,0,0,0,0,0,0
    1,0,0,0,0,0,0
    0,1,0,0,0,0,0
    0,0,1,0,0,0,0
    0,0,0,1,0,0,0
    0,0,0,0,1,0,0
    0,0,0,0,0,1,0
    0,0,0,0,0,0,1];

% Compute the syndrome of err
syn = zeros(1,3);
for i = 1:length(g(:,1))
    syn(1,i) = mod(sum(conj(err).* g(i,:)),2);     
end

% Find the correction corresponding to the syndrome
correctionRow = 1;
for i = 1:length(syn)
    correctionRow = correctionRow + 2^(3-i)*syn(i);
end

corr = MatRecovery(correctionRow,:);

Output = corr;

end

function Output = CorrectionFromSyn(syn) % used for measurments in the Z basis

% This function outputs a correction based on the measured syndrome using
% the lookup table

MatRecovery = [0,0,0,0,0,0,0
    1,0,0,0,0,0,0
    0,1,0,0,0,0,0
    0,0,1,0,0,0,0
    0,0,0,1,0,0,0
    0,0,0,0,1,0,0
    0,0,0,0,0,1,0
    0,0,0,0,0,0,1];

% Find the correction corresponding to the syndrome
correctionRow = 1;
for i = 1:length(syn)
    correctionRow = correctionRow + 2^(3-i)*syn(i);
end

corr = MatRecovery(correctionRow,:);

Output = corr;

end

function Output = ComputeLogicalUsingLookupTableX(SynX1,SynX2,SynX3,SynX4,Err3X,Err4X)

% This function outputs 1 if there is a logical X fault on the third or
% foruth block (or both).

% SynX1 is the X syndrome of the first block, SynX2 is the X syndrome of the second
% block, SynX3 is the X syndrome of the third block and SynX4 is the X
% syndrome of the fourth block

% Err3X and Err4X are the X errors on the third and fourth block

% Store error recovery's based on the measured syndromes
cxRow18 = CorrectionFromSyn(SynX1);
cxRow19 = CorrectionFromSyn(SynX2);
cxRow34 = CorrectionFromSyn(SynX3);
cxRow35 = CorrectionFromSyn(SynX4);

% Compute the final recovery using syndrome information from all four
% blocks
E1Final = mod(Err3X + cxRow18 + equivalenceTable(mod(cxRow18 + cxRow34,2)),2);
E3Final = mod(Err4X + cxRow18 + cxRow19 + equivalenceTable(mod(cxRow18 + cxRow19 + cxRow35,2)),2);

% Check if there is a logical X fault on the last two blocks
ef1Full = mod(sum(mod(E1Final + equivalenceTable(E1Final),2)),2);
ef3Full = mod(sum(mod(E3Final + equivalenceTable(E3Final),2)),2);

final = 0;

if (ef1Full == 1) || (ef3Full == 1)
    final = 1;
end

Output = final;

end

function Output = ComputeLogicalUsingLookupTableZ(SynZ1,SynZ2,SynZ3,SynZ4,Err3Z,Err4Z)

% This function outputs 1 if there is a logical Z fault on the third or
% foruth block (or both).

% SynZ1 is the Z syndrome of the first block, SynZ2 is the Z syndrome of the second
% block, SynZ3 is the Z syndrome of the third block and SynZ4 is the Z
% syndrome of the fourth block

% Err3Z and Err4Z are the Z errors on the third and fourth block

% Store measurment syndromes
czRow17 = CorrectionFromSyn(SynZ1);
czRow20 = CorrectionFromSyn(SynZ2);
czRow33 = CorrectionFromSyn(SynZ3);
czRow36 = CorrectionFromSyn(SynZ4);


% Compute the final recovery using syndrome information from all four
% blocks
E2Final = mod(Err3Z + czRow17 + czRow20 + equivalenceTable(mod(czRow17 + czRow20 + czRow33,2)),2);
E4Final = mod(Err4Z + czRow20 + equivalenceTable(mod(czRow20+ czRow36,2)),2);

% Check if there is a logical Z fault on the last two blocks
ef2Full = mod(sum(mod(E2Final + equivalenceTable(E2Final),2)),2);
ef4Full = mod(sum(mod(E4Final + equivalenceTable(E4Final),2)),2);

final = 0;

if (ef2Full == 1) || (ef4Full == 1)
    final = 1;
end

Output = final;

end