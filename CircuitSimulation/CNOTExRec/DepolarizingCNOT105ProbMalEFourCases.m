function DepolarizingCNOT105ProbMalE

% Circuit descriptors:
% -1: qubit non-active
% 0: Noiseless memory
% 1: Gate memory
% 2: Measurement memory
% 3: Preparation in X basis (|+> state)
% 4: Preparation in Z basis (|0> state)
% 5: Measurement in X basis
% 6: Measurement in Z basis
% 7: X gate
% 8: Z gate
% 10: H gate
% 11: S gate
% 20: T gate
%1---: Control qubit for CNOT with target ---
%1000: Target qubit for CNOT

% Circuit for CNOT 1-exRec
n = 7;
% Generate lookup table
lookUpO = lookUpOFunc(1,n);
lookUpPlus = lookUpPlusFunc(1,n);

%Test
% CNOT circuit containing all four EC blocks
CFull = [0,0,0,0,0,1000,1006,0,1010,0,0,0,0,0,1000,1006,0;
     4,1003,0,1000,2,1001,0,5,-1,4,1003,0,1000,2,1001,0,5;
     4,1000,6,-1,-1,-1,-1,-1,-1,4,1000,6,-1,-1,-1,-1,-1;
     4,1005,0,1002,5,-1,-1,-1,-1,4,1005,0,1002,5,-1,-1,-1;
     4,1000,6,-1,-1,-1,-1,-1,-1,4,1000,6,-1,-1,-1,-1,-1;
     3,1000,0,1008,2,0,1000,6,-1,3,1000,0,1008,2,0,1000,6;
     3,1006,5,-1,-1,-1,-1,-1,-1,3,1006,5,-1,-1,-1,-1,-1;
     3,1000,0,1000,6,-1,-1,-1,-1,3,1000,0,1000,6,-1,-1,-1;
     3,1008,5,-1,-1,-1,-1,-1,-1,3,1008,5,-1,-1,-1,-1,-1;
     0,0,0,0,0,1011,1000,0,1000,0,0,0,0,0,1011,1000,0;
     3,1000,0,1013,2,1000,0,6,-1,3,1000,0,1013,2,1000,0,6;
     3,1011,5,-1,-1,-1,-1,-1,-1,3,1011,5,-1,-1,-1,-1,-1;
     3,1000,0,1000,6,-1,-1,-1,-1,3,1000,0,1000,6,-1,-1,-1;
     3,1013,5,-1,-1,-1,-1,-1,-1,3,1013,5,-1,-1,-1,-1,-1;
     4,1016,0,1000,2,0,1010,5,-1,4,1016,0,1000,2,0,1010,5;
     4,1000,6,-1,-1,-1,-1,-1,-1,4,1000,6,-1,-1,-1,-1,-1;
     4,1018,0,1015,5,-1,-1,-1,-1,4,1018,0,1015,5,-1,-1,-1;
     4,1000,6,-1,-1,-1,-1,-1,-1,4,1000,6,-1,-1,-1,-1,-1];
 
 tic
parfor i = 1:18
    
       
    numIterations = 10^7;
    v = [8*10^-5,8.5*10^-5,9*10^-5,9.5*10^-5,10^-4,2*10^-4,3*10^-4,4*10^-4,5*10^-4,6*10^-4,7*10^-4,8*10^-4,9*10^-4,10^-3,1.5*10^-3,2*10^-3,2.5*10^-3,3*10^-3];
    errRate = v(1,i);
   
    errStatePrepString  = 'ErrorStatePrep';
    str_errRate = num2str(errRate,'%0.3e');
   % str_errStatePrep = strcat(errStatePrepString,str_errRate);
       
%     numIterations1 = 10^5;
%     
%     [eO,eOIndex,numAcceptedO] = PacceptErrorGeneratorOPrepUpper(errRate,numIterations1,n,lookUpPlus,lookUpO);
%     [ePlus,ePlusIndex,numAcceptedPlus] = PacceptErrorGeneratorPlusPrepUpper(errRate,numIterations1,n,lookUpPlus,lookUpO);
%     parsaveErrorStatePrep(str_errStatePrep,eO,eOIndex,numAcceptedO,ePlus,ePlusIndex,numAcceptedPlus);


switch i
    case 1
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep8.000e-05.mat');
    case 2
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep8.500e-05.mat');
    case 3
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep9.000e-05.mat');
    case 4
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep9.500e-05.mat');
    case 5
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep1.000e-04.mat');
    case 6
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep2.000e-04.mat');
    case 7
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep3.000e-04.mat');
    case 8
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep4.000e-04.mat');
    case 9
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep5.000e-04.mat');
    case 10
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep6.000e-04.mat');
    case 11
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep7.000e-04.mat');
    case 12
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep8.000e-04.mat');
    case 13
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep9.000e-04.mat');
    case 14
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep1.000e-03.mat');
    case 15
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep1.500e-03.mat');
    case 16
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep2.000e-03.mat');
    case 17
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep2.500e-03.mat');
    otherwise
        [eO,eOIndex,ePlus,ePlusIndex] = parload('ErrorStatePrep3.000e-03.mat');
end


if (~isempty(eOIndex)) && (~isempty(ePlusIndex))
    A = OutSynAndError(eO,eOIndex,ePlus,ePlusIndex,errRate,numIterations,CFull,n,lookUpPlus,lookUpO);        
    TempStr1 = 'SyndromeAndError';
    TempStr2 = '.txt';
    str_Final1 = strcat(TempStr1,str_errRate);
    str_Final2 = strcat(str_Final1,TempStr2);
    fid = fopen( str_Final2, 'w+t' );
    for ii = 1:size(A,1)
        fprintf(fid,'%g\t',A(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
end

end
toc

end

function parsaveErrorStatePrep(fname,eO,eOIndex,numAcceptedO,ePlus,ePlusIndex,numAcceptedPlus)
save(fname,'eO','eOIndex','numAcceptedO','ePlus','ePlusIndex','numAcceptedPlus');
end

function parsaveErrorVec(fname,errorVecMat)
save(fname,'errorVecMat');
end

function [out1,out2,out3,out4] = parload(fname)
load(fname);

out1 = eO;
out2 = eOIndex;
out3 = ePlus;
out4 = ePlusIndex;
end

function Output = lookUpOFunc(n,numQubits)

% Circuit for |0> state prep
lookUpO = zeros(19,5,numQubits);
C = [3,1007,1005,1003;
     3,1003,1006,1;
     4,1000,1,1000;
     3,1006,1,1005;
     4,0,1000,1000;
     4,1000,1000,1007;
     4,1000,1,1000];

% Find the number of gate storage locations in C
counterStorage = 0;
for i = 1:length(C(1,:))
    for j = 1:length(C(:,1))
        if C(j,i) == 1            
            counterStorage = counterStorage + 1;
        end
    end
end

% Find the number of CNOT locations in C
counterCNOT = 0;
for i = 1:length(C(1,:))
    for j = 1:length(C(:,1))
        if C(j,i) > 1000
            counterCNOT = counterCNOT + 1;
        end
    end
end

% fills in lookUpTable locations where lookUpO(i,j,k) has j = k = 1.
counter = 1;
for i = 1:counterStorage
    lookUpO(i,1,1) = 1;
    counter = counter + 1;
end

for i = 1 : length(C(:,1))
    lookUpO(counter,1,1) = C(i,1);
    counter = counter + 1;
end
 
for i = 1 : counterCNOT
    lookUpO(counter,1,1) = 1000;
    counter = counter + 1;
end

e = zeros(1,4);
% fills in LookUpO(i,j,:) for storage gate errors j = 2,3.
counter = 1;
for i = 2:length(C(1,:))
    for j = 1:length(C(:,1))
        for jj = 2:3
            if jj == 2
                if C(j,i) == 1
                    e(1,:) = [jj-1,j,1,i];
                    lookUpO(counter,jj,:) = xVector(C, n, e);                    
                end
            else
                if C(j,i) == 1
                    e(1,:) = [jj-1,j,1,i];
                    lookUpO(counter,jj,:) = zVector(C, n, e);
                    counter = counter + 1;
                end
            end            
        end        
    end
end

% fills in LookUpO(i,j,:) for state prep errors j = 2,3.
for i = 1:length(C(:,1))
    if C(i,1) == 3
        e(1,:) = [2,i,1,1];
        lookUpO(counter,3,:) = zVector(C, n, e);
        counter = counter + 1;
    else
        e(1,:) = [1,i,1,1];
        lookUpO(counter,2,:) = xVector(C, n, e);
        counter = counter + 1;
    end
end

% fills in LookUpO(i,j,:) for CNOT erros, j = 2,3.
for i = 2:length(C(1,:))
    for j = 1:length(C(:,1))
        for jj = [2,3,4,5]
            switch jj
                case 2
                    if C(j,i) > 1000
                        e(1,:) = [1,j,1,i];
                        lookUpO(counter,jj,:) = xVector(C, n, e);
                    end
                case 3
                    if C(j,i) > 1000
                        e(1,:) = [2,j,1,i];
                        lookUpO(counter,jj,:) = zVector(C, n, e);
                    end
                case 4
                    if C(j,i) > 1000                                               
                        e(1,:) = [4,j,1,i];                        
                        lookUpO(counter,jj,:) = xVector(C, n, e);                        
                    end
                otherwise
                    if C(j,i) > 1000                        
                        e(1,:) = [8,j,1,i];
                        lookUpO(counter,jj,:) = zVector(C, n, e);
                        counter = counter + 1;
                    end
            end
        end
    end
end

Output = lookUpO;

end

function Output = lookUpPlusFunc(n,numQubits)
% Circuit for |+> state prep
lookUpPlus = zeros(19,5,numQubits);
C = [4,1000,1000,1000;
     4,1000,1000,1;
     3,1002,1,1001;
     4,1000,1,1000;
     3,0,1001,1004;
     3,1004,1002,1000;
     3,1001,1,1006];

% Find the number of gate storage locations in C
counterStorage = 0;
for i = 1:length(C(1,:))
    for j = 1:length(C(:,1))
        if C(j,i) == 1
            counterStorage = counterStorage + 1;
        end
    end
end

% Find the number of CNOT locations in C
counterCNOT = 0;
for i = 1:length(C(1,:))
    for j = 1:length(C(:,1))
        if C(j,i) > 1000
            counterCNOT = counterCNOT + 1;
        end
    end
end

% fills in lookUpTable locations where lookUpO(i,j,k) has j = k = 1.
counter = 1;
for i = 1:counterStorage
    lookUpPlus(i,1,1) = 1;
    counter = counter + 1;
end

for i = 1 : length(C(:,1))
    lookUpPlus(counter,1,1) = C(i,1);
    counter = counter + 1;
end
 
for i = 1 : counterCNOT
    lookUpPlus(counter,1,1) = 1000;
    counter = counter + 1;
end

e = zeros(1,4);
% fills in LookUpO(i,j,:) for storage gate errors j = 2,3.
counter = 1;
for i = 2:length(C(1,:))
    for j = 1:length(C(:,1))
        for jj = 2:3
            if jj == 2
                if C(j,i) == 1
                    e(1,:) = [jj-1,j,1,i];
                    lookUpPlus(counter,jj,:) = xVector(C, n, e);                    
                end
            else
                if C(j,i) == 1
                    e(1,:) = [jj-1,j,1,i];
                    lookUpPlus(counter,jj,:) = zVector(C, n, e);
                    counter = counter + 1;
                end
            end            
        end        
    end
end

% fills in LookUpO(i,j,:) for state prep errors j = 2,3.
for i = 1:length(C(:,1))
    if C(i,1) == 3
        e(1,:) = [2,i,1,1];
        lookUpPlus(counter,3,:) = zVector(C, n, e);
        counter = counter + 1;
    else
        e(1,:) = [1,i,1,1];
        lookUpPlus(counter,2,:) = xVector(C, n, e);
        counter = counter + 1;
    end
end

% fills in LookUpO(i,j,:) for CNOT erros, j = 2,3.
for i = 2:length(C(1,:))
    for j = 1:length(C(:,1))
        for jj = [2,3,4,5]
            switch jj
                case 2
                    if C(j,i) > 1000
                        e(1,:) = [1,j,1,i];
                        lookUpPlus(counter,jj,:) = xVector(C, n, e);
                    end
                case 3
                    if C(j,i) > 1000
                        e(1,:) = [2,j,1,i];
                        lookUpPlus(counter,jj,:) = zVector(C, n, e);
                    end
                case 4
                    if C(j,i) > 1000                                               
                        e(1,:) = [4,j,1,i];                        
                        lookUpPlus(counter,jj,:) = xVector(C, n, e);                        
                    end
                otherwise
                    if C(j,i) > 1000                        
                        e(1,:) = [8,j,1,i];
                        lookUpPlus(counter,jj,:) = zVector(C, n, e);
                        counter = counter + 1;
                    end
            end
        end
    end
end

Output = lookUpPlus;
end

function Output = PropagationStatePrepArb(C, n, e)

% C encodes information about the circuit
% n is the number of qubits per encoded codeblock
% eN are the error entries (location and time)

% The Output matrix will contain (2q+m) rows and n columns. The paramters Output(i,j) are:
% i <= 2q and odd: Stores X type errors for output qubit #ceil(i/2) (order based on matrix C)
% i <= 2q and even: Stores Z type errors for output qubit #ceil(i/2) (order based on matrix C)
% i > 2q: Stores measurement error for measurement number #i-2q (type is X or Z depending on measurement type)

% The Errors matrix will be a tensor with 2 parameters Output(i, j)
% i: logical qubit number
% j<=n: X error at physical qubit number j
% j>n : Z error at physical qubit number j


% Error inputs eN are characterized by four parameters, thus a vector (i,j, k, l)
% i: error type: 0 = I,  1 = X, 2 = Z, 3 = Y
% j: logical qubit number (if the entry here is '0' then this will correspond to an identity "error")
% k: physical qubit number within a codeblock
% l: time location

N_meas = length(find(C==5)) + length(find(C==6)); % number of logical measurements in the circuit
N_output = length(C(:,end)) - sum(C(:,end)==-1) - sum(C(:,end)==5) - sum(C(:,end)==6); % number of output active qubits that have not been measured

Meas_dict = [find(C==5); find(C==6)];
Meas_dict = sort(Meas_dict);

Output = zeros(2*N_output + N_meas, n);
Errors = zeros(length(C(:,1)), 2*n);


for t= 1:length(C(1,:))
    
    % If the error occurs at a measurement location then the error is introduced before propagation of faults
    for j = 1:length(e(:,1))
        if e(j,1) ~= 0
            if e(j,4) == t && ( ( C(e(j,2),t) == 5 ) || ( C(e(j,2),t) == 6 ) )
                if e(j,1) == 1
                    Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                end
                if e(j,1) == 2
                    Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                end
                if e(j,1) == 3
                    Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                    Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                end
            end
        end
    end
    
    % Propagation of errors (do not need to do it for the first step of circuit
    if t>1
        for i = 1:length(C(:,t))
            if C(i, t) == 10
                % In this case must flip the X and Z error information
                v1 = Errors(i, 1:n);
                v2 = Errors(i, (n+1):end);
                Errors(i, 1:n) = v2;
                Errors(i, n+1:end) = v1;
            end
            
            if C(i, t) == 11
                v1 = Errors(i, 1:n);
                v2 = Errors(i, (n+1):end);
                Errors(i, 1:n) = mod(v1+v2, 2);
            end
            
            if C(i,t) > 1000
                % check to see if target qubit according to control qubit is actually a target qubit
                if C(C(i,t) - 1000, t) == 1000
                    v1 = Errors(i, 1:n);
                    v2 = Errors(i, (n+1):end);
                    w1 = Errors(C(i,t) - 1000, 1:n);
                    w2 = Errors(C(i,t) - 1000, (n+1):end);
                    %mod(v2+w2,2)
                    Errors(C(i,t) - 1000, 1:n) = mod(v1+w1, 2);
                    Errors(i, (n+1):end) = mod(v2+w2, 2);
                end
            end
            
            if C(i,t) == 5
                % This corresponds to X measurement, therefore need to look at Z errors
                find(Meas_dict==(i+(t-1)*length(C(:,1))) ); % Dont think these are needed, used for checking
                Output(2*N_output + find(Meas_dict==(i+(t-1)*length(C(:,1))) ), :) = Errors(i, (n+1):end);
                Errors(i, 1:end) = 0;
            end
            
            if C(i,t) == 6
                % This corresponds to Z measurement, therefore need to look at Z errors
                find(Meas_dict==(i+(t-1)*length(C(:,1))) ); % Dont think these are needed, used for checking
                Output(2*N_output + find(Meas_dict==(i+(t-1)*length(C(:,1))) ), :) = Errors(i, 1:n);
                Errors(i, 1:end) = 0;
            end
            
        end
    end
    
    % Introduce faults for locations that are not measurements
    for j = 1:length(e(:,1))
        if e(j,1) ~= 0
            if e(j,4) == t && ( C(e(j,2),t) ~= 5 ) && ( C(e(j,2),t) ~= 6 )
                % This IF statement checks to see if the gate at this location is NOT a CNOT or Prep
                if ( C(e(j,2),t) < 1000 ) %&& ( C(e(j,2),t) ~= 3 ) && ( C(e(j,2),t) ~= 4 )
                    if e(j,1) == 1
                        Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                    end
                    if e(j,1) == 2
                        Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                    end
                    if e(j,1) == 3
                        Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                        Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                    end
                end
                % Introduce errors in the case of CNOT gate for control and target qubits
                % Errors for control qubit are entry mod(e(j,1),4) according to standard indexing above
                if ( C(e(j,2),t) > 1000 )
                    if C(C(e(j,2),t) - 1000, t) == 1000
                        if mod(e(j,1),2) == 1
                            Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                        end
                        if mod(e(j,1),4) > 1
                            Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                        end
                        if mod(floor(e(j,1)/4),2) == 1
                            Errors(C(e(j,2),t) - 1000, e(j,3)) = mod(Errors(C(e(j,2),t) - 1000, e(j,3)) + 1, 2);
                        end
                        if mod(floor(e(j,1)/4),4) > 1
                            Errors(C(e(j,2),t) - 1000, e(j,3)+n) = mod(Errors(C(e(j,2),t) - 1000, e(j,3)+n) + 1, 2);
                        end
                    end
                end
            end
        end
    end
    
end

%Errors
counter = 1; % This will be used to iterate over the different qubits
for j = 1:length(C(:,end))
	if (C(j,end) ~= -1) && (C(j,end) ~= 5) &&  (C(j,end) ~= 6)
		Output(counter,:) = Errors(j,1:n);
		Output(counter+1,:) = Errors(j,(n+1):end);
		counter = counter + 2;
	end
end

end

function Output = xVector(C, n, e)

outputMatrix = PropagationStatePrepArb(C, n, e);
jx = 1;
for i = 1:length(outputMatrix)
    if mod(i,2) == 1
        xvector(jx) = outputMatrix(i,1);
        jx = jx + 1;
    end
end

Output = xvector;

end

function Output = zVector(C, n, e)

outputMatrix = PropagationStatePrepArb(C, n, e);
jz = 1;
for i = 1:length(outputMatrix)
    if mod(i,2) == 0
        zvector(jz) = outputMatrix(i,1);
        jz = jz + 1;
    end
end

Output = zvector;

end

function Output = PropagationArb(C, n, e, XPrepTable, ZPrepTable)

% C encodes information about the circuit
% n is the number of qubits per encoded codeblock
% eN are the error entries (location and time)

% The Output matrix will contain (2q+m) rows and n columns. The paramters Output(i,j) are:
% i <= 2q and odd: Stores X type errors for output qubit #ceil(i/2) (order based on matrix C)
% i <= 2q and even: Stores Z type errors for output qubit #ceil(i/2) (order based on matrix C)
% i > 2q: Stores measurement error for measurement number #i-2q (type is X or Z depending on measurement type)

% The Errors matrix will be a tensor with 2 parameters Output(i, j)
% i: logical qubit number
% j<=n: X error at physical qubit number j
% j>n : Z error at physical qubit number j


% Error inputs eN are characterized by four parameters, thus a vector (i,j, k, l)
% i: error type: 0 = I,  1 = X, 2 = Z, 3 = Y
% j: logical qubit number (if the entry here is '0' then this will correspond to an identity "error")
% k: physical qubit number within a codeblock
% l: time location

N_meas = length(find(C==5)) + length(find(C==6)); % number of logical measurements in the circuit
N_output = length(C(:,end)) - sum(C(:,end)==-1) - sum(C(:,end)==5) - sum(C(:,end)==6); % number of output active qubits that have not been measured

Meas_dict = [find(C==5); find(C==6)];
Meas_dict = sort(Meas_dict);

Output = zeros(2*N_output + N_meas, n);
Errors = zeros(length(C(:,1)), 2*n);



for t= 1:length(C(1,:))
    
    % If the error occurs at a measurement location then the error is introduced before propagation of faults
    for j = 1:length(e(:,1))
        if e(j,1) ~= 0
            if e(j,4) == t && ( ( C(e(j,2),t) == 5 ) || ( C(e(j,2),t) == 6 ) )
                if e(j,1) == 1
                    Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                end
                if e(j,1) == 2
                    Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                end
                if e(j,1) == 3
                    Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                    Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                end
            end
        end
    end
    
	% Propagation of errors (do not need to do it for the first step of
	% circuit)
	if t>1
		for i = 1:length(C(:,t))
			if C(i, t) == 10
				% In this case must flip the X and Z error information
				v1 = Errors(i, 1:n);
				v2 = Errors(i, (n+1):end);
				Errors(i, 1:n) = v2;
				Errors(i, n+1:end) = v1;
			end

			if C(i, t) == 11
				v1 = Errors(i, 1:n);
				v2 = Errors(i, (n+1):end);
				Errors(i, 1:n) = mod(v1+v2, 2);
			end

			if C(i,t) > 1000
				% check to see if target qubit according to control qubit is actually a target qubit
				if C(C(i,t) - 1000, t) == 1000
					v1 = Errors(i, 1:n);
					v2 = Errors(i, (n+1):end);
					w1 = Errors(C(i,t) - 1000, 1:n);
					w2 = Errors(C(i,t) - 1000, (n+1):end);
					%mod(v2+w2,2)
					Errors(C(i,t) - 1000, 1:n) = mod(v1+w1, 2);
					Errors(i, (n+1):end) = mod(v2+w2, 2);
				end
			end

			if C(i,t) == 5
				% This corresponds to X measurement, therefore need to look at Z errors
				find(Meas_dict==(i+(t-1)*length(C(:,1))) ); % Dont think these are needed, used for checking
				Output(2*N_output + find(Meas_dict==(i+(t-1)*length(C(:,1))) ), :) = Errors(i, (n+1):end);
				Errors(i, 1:end) = 0;
			end

			if C(i,t) == 6
				% This corresponds to Z measurement, therefore need to look at Z errors
				find(Meas_dict==(i+(t-1)*length(C(:,1))) ); % Dont think these are needed, used for checking
				Output(2*N_output + find(Meas_dict==(i+(t-1)*length(C(:,1))) ), :) = Errors(i, 1:n);
				Errors(i, 1:end) = 0;
			end

		end
	end
	
    % Introduce faults for locations that are not measurements
    for j = 1:length(e(:,1))
        if e(j,1) ~= 0
            if e(j,4) == t && ( C(e(j,2),t) ~= 5 ) && ( C(e(j,2),t) ~= 6 )
                % This IF statement checks to see if the gate at this location is NOT a CNOT or Prep
                if ( C(e(j,2),t) < 1000 ) && ( C(e(j,2),t) ~= 3 ) && ( C(e(j,2),t) ~= 4 )
                    if e(j,1) == 1
                        Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                    end
                    if e(j,1) == 2
                        Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                    end
                    if e(j,1) == 3
                        Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                        Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                    end
                end
                % Introduce errors in the case of CNOT gate for control and target qubits
                % Errors for control qubit are entry mod(e(j,1),4) according to standard indexing above
                if ( C(e(j,2),t) > 1000 )
                    if C(C(e(j,2),t) - 1000, t) == 1000
                        if mod(e(j,1),2) == 1
                            Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                        end
                        if mod(e(j,1),4) > 1
                            Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                        end
                        if mod(floor(e(j,1)/4),2) == 1
                            Errors(C(e(j,2),t) - 1000, e(j,3)) = mod(Errors(C(e(j,2),t) - 1000, e(j,3)) + 1, 2);
                        end
                        if mod(floor(e(j,1)/4),4) > 1
                            Errors(C(e(j,2),t) - 1000, e(j,3)+n) = mod(Errors(C(e(j,2),t) - 1000, e(j,3)+n) + 1, 2);
                        end
                    end
                end
                % Introduce errors in the case of |0> prep
                if ( C(e(j,2),t) == 4 )
                    eVec = zeros(1,n);
                    if mod(e(j,1),2) == 1
                        % Need to translate the entries in the Prep tables to right format
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 2, a);
                        end
                        Errors(e(j,2), 1:n ) = mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(e(j,1),4) > 1
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 3, a);
                        end
                        Errors(e(j,2), (n+1):end ) = mod(Errors(e(j,2), (n+1):end)+eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),2) == 1
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 4, a);
                        end
                        Errors(e(j,2), 1:n ) = mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),4) > 1
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 5, a);
                        end
                        Errors(e(j,2), (n+1):end ) = mod(Errors(e(j,2), (n+1):end) + eVec, 2);
                    end
                end
                % Introduce errors in the case of |+> prep
                if ( C(e(j,2),t) == 3 )
                    eVec = zeros(1,n);
                    if mod(e(j,1),2) == 1
                        % Need to translate the entries in the Prep tables to right format
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 2, a);
                        end
                        Errors(e(j,2), 1:n ) = mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(e(j,1),4) > 1
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 3, a);
                        end
                        Errors(e(j,2), (n+1):end ) = mod(Errors(e(j,2), (n+1):end)+eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),2) == 1
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 4, a);
                        end
                        Errors(e(j,2), 1:n ) = mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),4) > 1
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 5, a);
                        end
                        Errors(e(j,2), (n+1):end ) = mod(Errors(e(j,2), (n+1):end) + eVec, 2);
                    end
                end
            end
        end
    end


end

%Errors
counter = 1; % This will be used to iterate over the different qubits
for j = 1:length(C(:,end))
	if (C(j,end) ~= -1) && (C(j,end) ~= 5) &&  (C(j,end) ~= 6)
		Output(counter,:) = Errors(j,1:n);
		Output(counter+1,:) = Errors(j,(n+1):end);
		counter = counter + 2;
	end
end


end

function Output = equivalenceTable(errIn) % used for measurments in the Z basis
g1 = [0,0,0,1,1,1,1];
g2 = [0,1,1,0,0,1,1];
g3 = [1,0,1,0,1,0,1];

g = [g1;g2;g3];

syn = zeros(1,3);

for i = 1:3
    syn(1,i) = mod(sum(errIn.*g(i,:)), 2);
end

if isequal(syn,[0,0,0])
    corr = zeros(1,7);
elseif isequal(syn,[0,0,1])
    corr = [1,0,0,0,0,0,0];
elseif isequal(syn,[0,1,0])
    corr = [0,1,0,0,0,0,0];
elseif isequal(syn,[0,1,1])
    corr = [0,0,1,0,0,0,0];
elseif isequal(syn,[1,0,0])
    corr = [0,0,0,1,0,0,0];
elseif isequal(syn,[1,0,1])
    corr = [0,0,0,0,1,0,0];
elseif isequal(syn,[1,1,0])
    corr = [0,0,0,0,0,1,0];
elseif isequal(syn,[1,1,1])
    corr = [0,0,0,0,0,0,1];
end

Output = corr;

end

function Output = Syndrome(errIn)

g1 = [0,0,0,1,1,1,1];
g2 = [0,1,1,0,0,1,1];
g3 = [1,0,1,0,1,0,1];

g = [g1;g2;g3];

syn = zeros(1,3);

for i = 1:3
    syn(1,i) = mod( sum(errIn.*g(i,:)), 2);
end

Output = syn;

end

function Output = errorGeneratorPacceptOprepUpper(errRate)
% Generates errors in the ancilla preparation part of the EC's
e = zeros(1,4);
rows = 1;

% Locations within the state prep |0> circuit
for i = 1:4
    for l = 1:4
        xi = rand;
        if xi < errRate
            k = randi([1,3]);
            e(rows,:) = [k,i,l,1];
            rows = rows + 1;
        end
    end
    for l = 5:11
        if (l == 5) || (l == 6) || (l == 8) 
            xi = rand;
            if xi < 2*errRate/3
                e(rows,:) = [2,i,l,1];
                rows = rows + 1;
            end
        else
            xi = rand;
            if xi < 2*errRate/3
                e(rows,:) = [1,i,l,1];
                rows = rows + 1;
            end
        end
    end
    for l = 12:19
        xi = rand;
        if xi < errRate
            k = randi([1,15]);
            e(rows,:) = [k,i,l,1];
            rows = rows + 1;
        end
    end
end

for i = [1,3]
    for l = 1:7
        xi = rand;
        if xi < errRate
            k = randi([1,15]);
            e(rows,:) = [k,i,l,2];
            rows = rows + 1;
        end
    end
end

for i = [2,4]
    for l = 1:7
        xi = rand;
        if xi < 2*errRate/3
            e(rows,:) = [1,i,l,3];
            rows = rows + 1;
        end
    end
end

for l = 1:7
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(rows,:) = [k,3,l,4];
        rows = rows + 1;
    end
end

for l = 1:7
    xi = rand;
    if xi < 2*errRate/3
        e(rows,:) = [2,3,l,5];
        rows = rows + 1;
    end
end

Output = e;

end

function [Output1,Output2,Output3] = PacceptErrorGeneratorOPrepUpper(errRate,numIterations,n,lookUpPlus,lookUpO)

C = [4,1002,0,1000,2;
      4,1000,6,-1,-1;
      4,1004,0,1001,5;
      4,1000,6,-1,-1];

eAccepted = zeros(1,4);
eIndex = zeros(1,1);
countereIndex = 1;
counterRows = 0;
numAccepted = 0;

while length(eAccepted(:,1)) < numIterations
    
    e = errorGeneratorPacceptOprepUpper(errRate);
    
    if isequal(e,zeros(1,4))
        counter = 0;
    else
        counter = length(e(:,1));
    end
    
    propagatedMatrix = PropagationArb(C,n,e,lookUpPlus,lookUpO);
    if (sum(equivalenceTable(propagatedMatrix(3,:)))==0) && (sum(equivalenceTable(propagatedMatrix(4,:)))==0) && (sum(equivalenceTable(propagatedMatrix(5,:)))==0) && (mod(sum(propagatedMatrix(3,:)),2)==0) && (mod(sum(propagatedMatrix(4,:)),2)==0) && (mod(sum(propagatedMatrix(5,:)),2)==0)
        if counter ~= 0
            numAccepted = numAccepted + 1;
            counterLast = counterRows;
            counterRows = counterRows + counter;
            eAccepted((counterLast + 1):(counterLast + counter),:) = e;
            eIndex(countereIndex,1) = counterLast + 1;
            countereIndex = countereIndex + 1;
        else
            numAccepted = numAccepted + 1;
            counterLast = counterRows;
            counterRows = counterRows + 1;
            eAccepted((counterLast + 1),1:4) = e;
            eIndex(countereIndex,1) = counterLast + 1;
            countereIndex = countereIndex + 1;
        end
    end
    
end

Output1 = eAccepted;
Output2 = eIndex;
Output3 = numAccepted;

end

function Output = errorGeneratorPacceptPlusprepUpper(errRate)

e = zeros(1,4);
rows = 1;

for i = 1:4
    for l = 1:4
        xi = rand;
        if xi < errRate
            k = randi([1,3]);
            e(rows,:) = [k,i,l,1];
            rows = rows + 1;
        end
    end
    for l = 5:11
        if (l == 7) || (l == 9) || (l == 10) || (l == 11)
            xi = rand;
            if xi < 2*errRate/3
                e(rows,:) = [2,i,l,1];
                rows = rows + 1;
            end
        else
            xi = rand;
            if xi < 2*errRate/3
                e(rows,:) = [1,i,l,1];
                rows = rows + 1;
            end
        end
    end
    for l = 12:19
        xi = rand;
        if xi < errRate
            k = randi([1,15]);
            e(rows,:) = [k,i,l,1];
            rows = rows + 1;
        end
    end
end

for i = [2,4]
    for l = 1:7
        xi = rand;
        if xi < errRate
            k = randi([1,15]);
            e(rows,:) = [k,i,l,2];
            rows = rows + 1;
        end
    end
end

for i = [2,4]
    for l = 1:7
        xi = rand;
        if xi < 2*errRate/3
            e(rows,:) = [2,i,l,3];
            rows = rows + 1;
        end
    end
end

for l = 1:7
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(rows,:) = [k,1,l,4];
        rows = rows + 1;
    end
end

for l = 1:7
    xi = rand;
    if xi < 2*errRate/3
        e(rows,:) = [1,3,l,5];
        rows = rows + 1;
    end
end

Output = e;

end

function [Output1,Output2,Output3] = PacceptErrorGeneratorPlusPrepUpper(errRate,numIterations,n,lookUpPlus,lookUpO)

C = [3,1000,0,1003,2;
      3,1001,5,-1,-1;
      3,1000,0,1000,6;
      3,1003,5,-1,-1];

eAccepted = zeros(1,4);
eIndex = zeros(1,1);
countereIndex = 1;
counterRows = 0;
numAccepted = 0;

while length(eAccepted(:,1)) < numIterations
    
    e = errorGeneratorPacceptPlusprepUpper(errRate);
    
    if isequal(e,zeros(1,4))
        counter = 0;
    else
        counter = length(e(:,1));
    end
    
    propagatedMatrix = PropagationArb(C, n, e,lookUpPlus, lookUpO);
    if (sum(equivalenceTable(propagatedMatrix(3,:)))==0) && (sum(equivalenceTable(propagatedMatrix(4,:)))==0) && (sum(equivalenceTable(propagatedMatrix(5,:)))==0) && (mod(sum(propagatedMatrix(3,:)),2)==0) && (mod(sum(propagatedMatrix(4,:)),2)==0) && (mod(sum(propagatedMatrix(5,:)),2)==0)
        if counter ~= 0
            numAccepted = numAccepted + 1;
            counterLast = counterRows;
            counterRows = counterRows + counter;
            eAccepted((counterLast + 1):(counterLast + counter),:) = e;
            eIndex(countereIndex,1) = counterLast + 1;
            countereIndex = countereIndex + 1;
        else
            numAccepted = numAccepted + 1;
            counterLast = counterRows;
            counterRows = counterRows + 1;
            eAccepted((counterLast + 1),1:4) = e;
            eIndex(countereIndex,1) = counterLast + 1;
            countereIndex = countereIndex + 1;
        end
    end
    
end

Output1 = eAccepted;
Output2 = eIndex;
Output3 = numAccepted;

end

function Output = errorGeneratorRemaining(errRate)
% generates remaining errors outside state-prep circuits. 
e = zeros(1,4);
counter = 1;

% Errors in first block

for i = [2,6] % Storage locations in first block
    for l = 1:7
        xi = rand;
        if xi < errRate
            k = randi([1,3]);
            e(counter,:) = [k,i,l,5];
            counter = counter + 1;
        end
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,2,l,6];
        counter = counter + 1;
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,1,l,7];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in X basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [2,2,l,8];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in Z basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [1,6,l,8];
        counter = counter + 1;
    end
end

% Error at encoded CNOT location
for l = 1:7
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,1,l,9];
        counter = counter + 1;
    end
end

% Error in second block (connected to first data qubit)

for i = [2,6] % Storage locations in first block
    for l = 1:7
        xi = rand;
        if xi < errRate
            k = randi([1,3]);
            e(counter,:) = [k,i,l,14];
            counter = counter + 1;
        end
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,2,l,15];
        counter = counter + 1;
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,1,l,16];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in X basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [2,2,l,17];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in Z basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [1,6,l,17];
        counter = counter + 1;
    end
end

% Errors in third block (connected to second data qubit)

for i = [11,15] % Storage locations in first block
    for l = 1:7
        xi = rand;
        if xi < errRate
            k = randi([1,3]);
            e(counter,:) = [k,i,l,5];
            counter = counter + 1;
        end
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,10,l,6];
        counter = counter + 1;
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,15,l,7];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in Z basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [1,11,l,8];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in X basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [2,15,l,8];
        counter = counter + 1;
    end
end

% Errors in fourth block
for i = [11,15] % Storage locations in first block
    for l = 1:7
        xi = rand;
        if xi < errRate
            k = randi([1,3]);
            e(counter,:) = [k,i,l,14];
            counter = counter + 1;
        end
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,10,l,15];
        counter = counter + 1;
    end
end
for l = 1:7 % CNOT locations
    xi = rand;
    if xi < errRate
        k = randi([1,15]);
        e(counter,:) = [k,15,l,16];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in Z basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [1,11,l,17];
        counter = counter + 1;
    end
end
for l = 1:7 % measurment in X basis
    xi = rand;
    if xi < 2*errRate/3
        e(counter,:) = [2,15,l,17];
        counter = counter + 1;
    end
end

Output = e;

end

function [Output1,Output2,Output3,Output4] = depolarizingGenerator(eO,eOIndex,ePlus,ePlusIndex,errRate,numIterations,CFull,CFirst,n,lookUpPlus,lookUpO)
errorVec1 = zeros(1,15);
errorVec2 = zeros(1,15);
errorVec3 = zeros(1,15);
errorVec4 = zeros(1,15);
for ii = 1:numIterations
    % Generate errors from |0> state-prep circuit in first 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    
    e = errorO;
    rows = length(e(:,1));
    
    % Genrate errors from |+> state-prep circuit in first 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);

    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    
    e1Plus = errorPlus;
    rowse1Plus = length(e1Plus(:,1));
    e((rows + 1):(rows + rowse1Plus),:) = e1Plus;
    rows = rows + rowse1Plus;
    
    
    % Generate errors from |0> state-prep circuit in the second 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    
    e2O = errorO;
    for i = 1:length(e2O(:,1))
        e2O(i,2) = e2O(i,2) + 13;
    end
    rowse2O = length(e2O(:,1));
    e((rows + 1):(rows + rowse2O),:) = e2O;
    rows = rows + rowse2O;
    
    
    % Generate errors from |+> state-prep in the second 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);
    
    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    e2Plus = errorPlus;
    for i = 1:length(e2Plus(:,1))
        e2Plus(i,2) = e2Plus(i,2) + 5;
    end
    rowse2Plus = length(e2Plus(:,1));
    e((rows + 1):(rows + rowse2Plus),:) = e2Plus;
    rows = rows + rowse2Plus;    
    
    % Generate errors from |0> state-prep circuit in third 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    e3O = errorO;
    for i = 1:length(e3O(:,1))
        e3O(i,4) = e3O(i,4) + 9;
    end
    rowse3O = length(e3O(:,1));
    e((rows + 1):(rows + rowse3O),:) = e3O;
    rows = rows + rowse3O;
    
    % Genrate errors from |+> state-prep circuit in third 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);
    
    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    e3Plus = errorPlus;
    for i = 1:length(e3Plus(:,1))
        e3Plus(i,4) = e3Plus(i,4) + 9;
    end
    rowse3Plus = length(e3Plus(:,1));
    e((rows + 1):(rows + rowse3Plus),:) = e3Plus;
    rows = rows + rowse3Plus;
    
    
    % Generate errors from |0> state-prep circuit in the fourth 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    e4O = errorO;
    for i = 1:length(e4O(:,1))
        e4O(i,2) = e4O(i,2) + 13;
        e4O(i,4) = e4O(i,4) + 9;
    end
    rowse4O = length(e4O(:,1));
    e((rows + 1):(rows + rowse4O),:) = e4O;
    rows = rows + rowse4O;
    
    % Generate errors from |+> state-prep in the fourth 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);
    
    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    e4Plus = errorPlus;
    for i = 1:length(e4Plus(:,1))
        e4Plus(i,2) = e4Plus(i,2) + 5;
        e4Plus(i,4) = e4Plus(i,4) + 9;
    end
    rowse4Plus = length(e4Plus(:,1));
    e((rows + 1):(rows + rowse4Plus),:) = e4Plus;
    rows = rows + rowse4Plus;    
    
    % Next we generate a random error in the circuit (outside of state-prep
    % circuits) to add to the matrix e (given the error rate errRate).
    
    eRemain = errorGeneratorRemaining(errRate);
    
    e((rows + 1):(rows + length(eRemain(:,1))),:) = eRemain;    
   
    Cint = PropagationArb(CFirst,n,e,lookUpPlus,lookUpO); % Propagate errors up to one time step past the encoded CNOT
    Cfinal = PropagationArb(CFull,n,e,lookUpPlus,lookUpO); % Propagate errors through the full CNOT circuit
    
    % Store measurment syndromes
    czRow17 = equivalenceTable(Cfinal(17,1:n));
    cxRow18 = equivalenceTable(Cfinal(18,1:n));
    cxRow19 = equivalenceTable(Cfinal(19,1:n));
    czRow20 = equivalenceTable(Cfinal(20,1:n));
    czRow33 = equivalenceTable(Cfinal(33,1:n));
    cxRow34 = equivalenceTable(Cfinal(34,1:n));
    cxRow35 = equivalenceTable(Cfinal(35,1:n));
    czRow36 = equivalenceTable(Cfinal(36,1:n));
    
    % Case where both TEC's have been removed
    CBothECRemoved(1,:) = mod(Cint(1,:) + cxRow18,2);
    CBothECRemoved(2,:) = mod(Cint(2,:) + czRow17,2);
    CBothECRemoved(3,:) = mod(Cint(3,:) + cxRow19,2);
    CBothECRemoved(4,:) = mod(Cint(4,:) + czRow20,2);
    
    ef1BothECRemoved = mod(sum(mod(CBothECRemoved(1,:) + equivalenceTableX(CBothECRemoved(1,:)),2)),2);
    ef2BothECRemoved = mod(sum(mod(CBothECRemoved(2,:) + equivalenceTableZ(CBothECRemoved(2,:)),2)),2);   
    ef3BothECRemoved = mod(sum(mod(CBothECRemoved(3,:) + equivalenceTableX(CBothECRemoved(3,:)),2)),2);
    ef4BothECRemoved = mod(sum(mod(CBothECRemoved(4,:) + equivalenceTableZ(CBothECRemoved(4,:)),2)),2);
    
    stringVec1 = [ef1BothECRemoved,ef2BothECRemoved,ef3BothECRemoved,ef4BothECRemoved];
    
    if (sum(stringVec1) ~= 0)
        ind1 = (2^3)*ef4BothECRemoved + (2^2)*ef3BothECRemoved + 2*ef2BothECRemoved + ef1BothECRemoved;
        errorVec1(1,ind1) = errorVec1(1,ind1) + 1;
    end
    
    % Case where both TEC's are kept    
    CFullMat(1,:) = mod(Cfinal(1,:) + cxRow18 + multilayerDecoderX(mod(cxRow18 + cxRow34,2)),2);    
    CFullMat(2,:) = mod(Cfinal(2,:) + czRow17 + czRow20 + multilayerDecoderZ(mod(czRow17 + czRow20 + czRow33,2)),2);
    CFullMat(3,:) = mod(Cfinal(3,:) + cxRow18 + cxRow19 + multilayerDecoderX(mod(cxRow18 + cxRow19 + cxRow35,2)),2);    
    CFullMat(4,:) = mod(Cfinal(4,:) + czRow20 + multilayerDecoderZ(mod(czRow20+ czRow36,2)),2);
    
    ef1Full = mod(sum(mod(CFullMat(1,:) + multilayerDecoderX(CFullMat(1,:)),2)),2);
    ef2Full = mod(sum(mod(CFullMat(2,:) + multilayerDecoderZ(CFullMat(2,:)),2)),2);   
    ef3Full = mod(sum(mod(CFullMat(3,:) + multilayerDecoderX(CFullMat(3,:)),2)),2);
    ef4Full = mod(sum(mod(CFullMat(4,:) + multilayerDecoderZ(CFullMat(4,:)),2)),2);
    
    stringVec2 = [ef1Full,ef2Full,ef3Full,ef4Full];
    
    if (sum(stringVec2) ~= 0)
        ind2 = (2^3)*ef4Full + (2^2)*ef3Full + 2*ef2Full + ef1Full;
        errorVec2(1,ind2) = errorVec2(1,ind2) + 1;
    end
    
    % Case where control TEC is removed
    CConECRem(1,:) = mod(Cint(1,:) + cxRow18,2);    
    CConECRem(2,:) = mod(Cint(2,:) + czRow17 + czRow20 + multilayerDecoderZ(mod(czRow17 + czRow20,2)),2);
    CConECRem(3,:) = mod(Cfinal(3,:) + cxRow18 + cxRow19 + multilayerDecoderX(mod(cxRow18 + cxRow19 + cxRow35,2)),2);    
    CConECRem(4,:) = mod(Cfinal(4,:) + czRow20 + multilayerDecoderZ(mod(czRow20+ czRow36,2)),2);
    
    ef1ConECRem = mod(sum(mod(CConECRem(1,:) + multilayerDecoderX(CConECRem(1,:)),2)),2);
    ef2ConECRem = mod(sum(mod(CConECRem(2,:) + multilayerDecoderZ(CConECRem(2,:)),2)),2);   
    ef3ConECRem = mod(sum(mod(CConECRem(3,:) + multilayerDecoderX(CConECRem(3,:)),2)),2);
    ef4ConECRem = mod(sum(mod(CConECRem(4,:) + multilayerDecoderZ(CConECRem(4,:)),2)),2);
    
    stringVec3 = [ef1ConECRem,ef2ConECRem,ef3ConECRem,ef4ConECRem];
    
    if (sum(stringVec3) ~= 0)
        ind3 = (2^3)*ef4ConECRem + (2^2)*ef3ConECRem + 2*ef2ConECRem + ef1ConECRem;
        errorVec3(1,ind3) = errorVec3(1,ind3) + 1;
    end
    
    % Case where target TEC is removed
    CTarECRem(1,:) = mod(Cfinal(1,:) + cxRow18 + multilayerDecoderX(mod(cxRow18 + cxRow34,2)),2);    
    CTarECRem(2,:) = mod(Cfinal(2,:) + czRow17 + czRow20 + multilayerDecoderZ(mod(czRow17 + czRow20 + czRow33,2)),2);
    CTarECRem(3,:) = mod(Cint(3,:) + cxRow18 + cxRow19 + multilayerDecoderX(mod(cxRow18 + cxRow19,2)),2);    
    CTarECRem(4,:) = mod(Cint(4,:) + czRow20,2);
    
    ef1TarECRem = mod(sum(mod(CTarECRem(1,:) + multilayerDecoderX(CTarECRem(1,:)),2)),2);
    ef2TarECRem = mod(sum(mod(CTarECRem(2,:) + multilayerDecoderZ(CTarECRem(2,:)),2)),2);   
    ef3TarECRem = mod(sum(mod(CTarECRem(3,:) + multilayerDecoderX(CTarECRem(3,:)),2)),2);
    ef4TarECRem = mod(sum(mod(CTarECRem(4,:) + multilayerDecoderZ(CTarECRem(4,:)),2)),2);
    
    stringVec4 = [ef1TarECRem,ef2TarECRem,ef3TarECRem,ef4TarECRem];
    
    if (sum(stringVec4) ~= 0)        
        ind4 = (2^3)*ef4TarECRem + (2^2)*ef3TarECRem + 2*ef2TarECRem + ef1TarECRem;
        errorVec4(1,ind4) = errorVec4(1,ind4) + 1;
    end
    
end

Output1 = errorVec1;
Output2 = errorVec2;
Output3 = errorVec3;
Output4 = errorVec4;

end

function Output = OutSynAndError(eO,eOIndex,ePlus,ePlusIndex,errRate,numIterations,CFull,n,lookUpPlus,lookUpO)

MatFinal = zeros(4*numIterations,20); % Matrix containing the X and Z syndrome (6 bits) along with the error (14 bits). Format : (XSyn|ZSyn,XErr|ZErr)

for ii = 1:numIterations
    % Generate errors from |0> state-prep circuit in first 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    
    e = errorO;
    rows = length(e(:,1));
    
    % Genrate errors from |+> state-prep circuit in first 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);
    
    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    
    e1Plus = errorPlus;
    rowse1Plus = length(e1Plus(:,1));
    e((rows + 1):(rows + rowse1Plus),:) = e1Plus;
    rows = rows + rowse1Plus;
    
    
    % Generate errors from |0> state-prep circuit in the second 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    
    e2O = errorO;
    for i = 1:length(e2O(:,1))
        e2O(i,2) = e2O(i,2) + 13;
    end
    rowse2O = length(e2O(:,1));
    e((rows + 1):(rows + rowse2O),:) = e2O;
    rows = rows + rowse2O;
    
    
    % Generate errors from |+> state-prep in the second 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);
    
    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    e2Plus = errorPlus;
    for i = 1:length(e2Plus(:,1))
        e2Plus(i,2) = e2Plus(i,2) + 5;
    end
    rowse2Plus = length(e2Plus(:,1));
    e((rows + 1):(rows + rowse2Plus),:) = e2Plus;
    rows = rows + rowse2Plus;
    
    % Generate errors from |0> state-prep circuit in third 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    e3O = errorO;
    for i = 1:length(e3O(:,1))
        e3O(i,4) = e3O(i,4) + 9;
    end
    rowse3O = length(e3O(:,1));
    e((rows + 1):(rows + rowse3O),:) = e3O;
    rows = rows + rowse3O;
    
    % Genrate errors from |+> state-prep circuit in third 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);
    
    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    e3Plus = errorPlus;
    for i = 1:length(e3Plus(:,1))
        e3Plus(i,4) = e3Plus(i,4) + 9;
    end
    rowse3Plus = length(e3Plus(:,1));
    e((rows + 1):(rows + rowse3Plus),:) = e3Plus;
    rows = rows + rowse3Plus;
    
    
    % Generate errors from |0> state-prep circuit in the fourth 1-EC
    lORand = randi([1,length(eOIndex(:,1))]);
    
    if eOIndex(lORand,1) ~= 0
        indexStart = eOIndex(lORand,1);
        if lORand == length(eOIndex(:,1))
            indexEnd = length(eO(:,1));
        else
            indexEnd = eOIndex(lORand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorO = eO(indexStart:indexEnd,:);
    
    for i = 1:length(errorO(:,1))
        errorO(i,2) = errorO(i,2) + 1; % Increase index of column 2 by one so that error is inserted in the correct row in the full circuit.
    end
    e4O = errorO;
    for i = 1:length(e4O(:,1))
        e4O(i,2) = e4O(i,2) + 13;
        e4O(i,4) = e4O(i,4) + 9;
    end
    rowse4O = length(e4O(:,1));
    e((rows + 1):(rows + rowse4O),:) = e4O;
    rows = rows + rowse4O;
    
    % Generate errors from |+> state-prep in the fourth 1-EC
    lPlusRand = randi([1,length(ePlusIndex(:,1))]);
    
    if ePlusIndex(lPlusRand,1) ~= 0
        indexStart = ePlusIndex(lPlusRand,1);
        if lPlusRand == length(ePlusIndex(:,1))
            indexEnd = length(ePlus(:,1));
        else
            indexEnd = ePlusIndex(lPlusRand + 1,1) - 1;
        end
    else
        indexStart = 1;
        indexEnd = 1;
    end
    
    errorPlus = ePlus(indexStart:indexEnd,:);
    
    for i = 1:length(errorPlus(:,1))
        errorPlus(i,2) = errorPlus(i,2) + 5;
    end
    e4Plus = errorPlus;
    for i = 1:length(e4Plus(:,1))
        e4Plus(i,2) = e4Plus(i,2) + 5;
        e4Plus(i,4) = e4Plus(i,4) + 9;
    end
    rowse4Plus = length(e4Plus(:,1));
    e((rows + 1):(rows + rowse4Plus),:) = e4Plus;
    rows = rows + rowse4Plus;
    
    % Next we generate a random error in the circuit (outside of state-prep
    % circuits) to add to the matrix e (given the error rate errRate).
    
    eRemain = errorGeneratorRemaining(errRate);
    
    e((rows + 1):(rows + length(eRemain(:,1))),:) = eRemain;
    
    Cfinal = PropagationArb(CFull,n,e,lookUpPlus,lookUpO); % Propagate errors through the full CNOT circuit
    
    % Store measurment syndromes
    ErrZ17 = Cfinal(17,1:n);
    SynZ17 = Syndrome(ErrZ17);
    
    ErrX18 = Cfinal(18,1:n);
    SynX18 = Syndrome(ErrX18);
    
    ErrX19 = Cfinal(19,1:n);
    SynX19 = Syndrome(ErrX19);
    
    ErrZ20 = Cfinal(20,1:n);
    SynZ20 = Syndrome(ErrZ20);
    
    ErrZ33 = Cfinal(33,1:n);
    SynZ33 = Syndrome(ErrZ33);
    
    ErrX34 = Cfinal(34,1:n);
    SynX34 = Syndrome(ErrX34);
    
    ErrX35 = Cfinal(35,1:n);
    SynX35 = Syndrome(ErrX35);
    
    ErrZ36 = Cfinal(36,1:n);
    SynZ36 = Syndrome(ErrZ36);
    
    MatTemp = [SynX18,SynZ17,ErrX18,ErrZ17;SynX19,SynZ20,ErrX19,ErrZ20;SynX34,SynZ33,ErrX34,ErrZ33;SynX35,SynZ36,ErrX35,ErrZ36];
    
    MatFinal((4*(ii-1)+1):(4*ii),:) = MatTemp;
    
    
end

Output = MatFinal;

end