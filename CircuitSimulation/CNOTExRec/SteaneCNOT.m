function SteaneCNOT

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
    errRates = [1*10^-3, 1*10^-4, 1.5*10^-3, ...
               2*10^-3, 2*10^-4, 2.5*10^-3, 3*10^-3, ...
               3*10^-4, 4*10^-4, 5*10^-4, 6*10^-4, ...
               7*10^-4, 8*10^-4, 8*10^-5, 8.5*10^-5, ...
               9*10^-4, 9*10^-5, 9.5*10^-5];
    numIterations = 10^5;
    
    global g
    g= [0, 0, 0, 1, 1, 1, 1;
        0, 1, 1, 0, 0, 1, 1;
        1, 0, 1, 0, 1, 0, 1];

    % Generate lookup table
    load('Circuits')
    lookUpO = lookUp(1, n, Circuits.ZeroStatePrepCircuit);
    lookUpPlus = lookUp(1, n, Circuits.PlusStatePrepCircuit);

    for i= 1:length(errRates)

        errRate= errRates(i);
        outstream = strcat('Err', num2str(errRate,'%0.1e'), ...
            'Len', num2str(numIterations,'%0.1e')); 
        load(strcat('ErrorStatePrep', num2str(errRate,'%0.3e'), '.mat'));
        dlmwrite(outstream, [], 'delimiter', '')
        FindSyndromeAndError(eO, eOIndex, ePlus, ePlusIndex, ...
            errRate, numIterations, CFull, n, ...
            lookUpPlus, lookUpO, outstream);

    end
    
end

function result = lookUp(n, numQubits, C)

    % Circuit for |0> state prep
    % 19 = num gate storage (4) + num CNOTS (8) + num qubits (7) 
    result = zeros(19, 5, numQubits);

    % Find the number of gate storage locations in C
    numStorage = sum(C(:) == 1);

    % fills in lookUpTable locations where lookUpO(i,j,k) has j = k = 1.
    result(1:numStorage, 1, 1) = 1;
    result(numStorage + 1: numStorage + length(C), 1, 1)= C(:, 1);
    result(numStorage + length(C) + 1: end, 1, 1)= 1000;

    % fills in LookUpO(i,j,:) for storage gate errors j = 2,3.
    counter = 1;
    for i = 2:size(C, 2)
        for j = 1:size(C, 1)
            if C(j,i) == 1
                result(counter, 2, :) = xVector(C, n, [1,j,1,i]);                    
                result(counter, 3, :) = zVector(C, n, [2,j,1,i]);
                counter = counter + 1;
            end
        end
    end

    % fills in LookUpO(i,j,:) for state prep errors j = 2,3.
    for i = 1:length(C(:,1))
        if C(i,1) == 3
            result(counter, 3, :) = zVector(C, n, [2,i,1,1]);
            counter = counter + 1;
        else
            result(counter, 2, :) = xVector(C, n, [1,i,1,1]);
            counter = counter + 1;
        end
    end

    % fills in LookUpO(i,j,:) for CNOT erros, j = 2,3.
    for i = 2:length(C(1,:))
        for j = 1:length(C(:,1))
            if C(j,i) > 1000
                result(counter, 2, :) = xVector(C, n, [1,j,1,i]);
                result(counter, 3, :) = zVector(C, n, [2,j,1,i]);
                result(counter, 4, :) = xVector(C, n, [4,j,1,i]);                        
                result(counter, 5, :) = zVector(C, n, [8,j,1,i]);
                counter = counter + 1;
            end
        end
    end

end

function result = xVector(C, n, e)
    outputMatrix = PropagationStatePrepArb(C, n, e);
    result = outputMatrix(1:2:end);
end

function result = zVector(C, n, e)
    outputMatrix = PropagationStatePrepArb(C, n, e);
    result = outputMatrix(2:2:end);
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

function FindSyndromeAndError(eO, eOIndex, ePlus, ePlusIndex, ...
    errRate, numIterations, CFull, n, lookUpPlus, lookUpO, outstream)
    
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

        % Increase index of column 2 by 1 so that error is inserted in the 
        % correct row in the full circuit.
        for i = 1:length(errorO(:,1))
            errorO(i,2) = errorO(i,2) + 1;
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

        % Increase index of column 2 by 1 so that error is inserted in the
        % correct row in the full circuit.
        for i = 1:length(errorO(:,1))
            errorO(i,2) = errorO(i,2) + 1;
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

        % Increase index of column 2 by 1 so that error is inserted in the 
        % correct row in the full circuit.
        for i = 1:length(errorO(:,1))
            errorO(i,2) = errorO(i,2) + 1;
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

        % Increase index of column 2 by 1 so that error is inserted in the 
        % correct row in the full circuit.
        for i = 1:length(errorO(:,1))
            errorO(i,2) = errorO(i,2) + 1; 
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

        % Generate a random error in the circuit (outside of state-prep
        % circuits) to add to the matrix e (given the error rate errRate).
        eRemain = errorGeneratorRemaining(errRate);    
        e((rows + 1):(rows + length(eRemain(:,1))),:) = eRemain;

        % Propagate errors through the full CNOT circuit
        Cfinal = PropagationArb(CFull,n,e,lookUpPlus,lookUpO);

        % Store Z errors and syndromes
        % ErrZ17 = Cfinal(17,1:n);
        % SynZ17 = Syndrome(ErrZ17);
        % ErrZ20 = Cfinal(20,1:n);
        % SynZ20 = Syndrome(ErrZ20);
        % ErrZ33 = Cfinal(33,1:n);
        % SynZ33 = Syndrome(ErrZ33);
        % ErrZ36 = Cfinal(36,1:n);
        % SynZ36 = Syndrome(ErrZ36);

        % Store X errors and syndromes
        ErrX18 = Cfinal(18,1:n);
        SynX18 = Syndrome(ErrX18);
        ErrX19 = Cfinal(19,1:n);
        SynX19 = Syndrome(ErrX19);
        ErrX34 = Cfinal(34,1:n);
        SynX34 = Syndrome(ErrX34);
        ErrX35 = Cfinal(35,1:n);
        SynX35 = Syndrome(ErrX35);

        SyndromeDataX = [SynX18, SynX19, SynX34, SynX35];
        RecoveryDataX = [ErrX34, ErrX35];

        if (any(SyndromeDataX))
            LookUpFail= ComputeLogicalUsingLookupTableX(...
                SynX18, SynX19, SynX34, SynX35, ErrX34, ErrX35); 
            dlmwrite(outstream, [...
                SyndromeDataX, RecoveryDataX, LookUpFail], ...
                '-append', 'delimiter', '')
        end

    end

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

function Output = PropagationArb(C, n, e, XPrepTable, ZPrepTable)

% C encodes information about the circuit
% n is the number of qubits per encoded codeblock
% eN are the error entries (location and time)

% The Output matrix will contain (2q+m) rows and n columns. 
% The paramters Output(i,j) are:
% 
% i <= 2q and odd: Stores X type errors for 
% output qubit #ceil(i/2) (order based on matrix C)
% 
% i <= 2q and even: Stores Z type errors for 
% output qubit #ceil(i/2) (order based on matrix C)
% 
% i > 2q: Stores measurement error for 
% measurement number #i-2q (type is X or Z depending on measurement type)
% 
% The Errors matrix will be a tensor with 2 parameters Output(i, j)
% i: logical qubit number
% j<=n: X error at physical qubit number j
% j>n : Z error at physical qubit number j


% Error inputs eN are characterized by four parameters, 
% thus a vector (i,j,k,l) where
% i: error type: 0 = I,  1 = X, 2 = Z, 3 = Y
% j: logical qubit number 
%    (if the entry here is '0' it corresponds to an identity "error")
% k: physical qubit number within a codeblock
% l: time location

% number of logical measurements in the circuit
N_meas = length(find(C==5)) + length(find(C==6)); 
% number of output active qubits that have not been measured
N_output = length(C(:,end)) - sum(C(:,end)==-1) ...
           - sum(C(:,end)==5) - sum(C(:,end)==6);

Meas_dict = [find(C==5); find(C==6)];
Meas_dict = sort(Meas_dict);

Output = zeros(2*N_output + N_meas, n);
Errors = zeros(length(C(:,1)), 2*n);

for t= 1:length(C(1,:))
    % If the error occurs at a measurement location then 
    % the error is introduced before propagation of faults
    for j = 1:length(e(:,1))
        if e(j,1) ~= 0
            if e(j,4) == t && ( ( C(e(j,2),t) == 5)||(C(e(j,2),t) == 6 ) )
                if e(j,1) == 1
                    Errors(e(j,2), e(j,3)) = ...
                        mod(Errors(e(j,2), e(j,3)) + 1, 2);
                end
                if e(j,1) == 2
                    Errors(e(j,2), e(j,3)+n) = ...
                        mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                end
                if e(j,1) == 3
                    Errors(e(j,2), e(j,3)) = ...
                        mod(Errors(e(j,2), e(j,3)) + 1, 2);
                    Errors(e(j,2), e(j,3)+n) = ...
                        mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
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
				% check to see if target qubit according to control qubit
                % is actually a target qubit
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
				% This corresponds to X measurement, 
                % therefore need to look at Z errors
				
                % Dont think these are needed, used for checking
                find(Meas_dict==(i+(t-1)*length(C(:,1))) ); 

                Output(2*N_output + ...
                    find(Meas_dict==(i+(t-1)*length(C(:,1))) ), :) ...
                         = Errors(i, (n+1):end);
				Errors(i, 1:end) = 0;
			end

			if C(i,t) == 6
				% This corresponds to Z measurement, 
                % therefore need to look at Z errors
				
                % Dont think these are needed, used for checking
                find(Meas_dict==(i+(t-1)*length(C(:,1))) );
                
                Output(2*N_output + ...
                    find(Meas_dict==(i+(t-1)*length(C(:,1))) ), :) ...
                         = Errors(i, 1:n);
				Errors(i, 1:end) = 0;
			end

		end
	end
	
    % Introduce faults for locations that are not measurements
    for j = 1:length(e(:,1))
        if e(j,1) ~= 0
            if e(j,4) == t && ( C(e(j,2),t) ~= 5 ) && ( C(e(j,2),t) ~= 6 )
                % This IF statement checks to see if the gate
                % at this location is NOT a CNOT or Prep
                if ( C(e(j,2),t) < 1000 ) ...
                        && ( C(e(j,2),t) ~= 3 ) && ( C(e(j,2),t) ~= 4 )
                    if e(j,1) == 1
                        Errors(e(j,2), e(j,3)) = ...
                            mod(Errors(e(j,2), e(j,3)) + 1, 2);
                    end
                    if e(j,1) == 2
                        Errors(e(j,2), e(j,3)+n) = ...
                            mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                    end
                    if e(j,1) == 3
                        Errors(e(j,2), e(j,3)) = ...
                            mod(Errors(e(j,2), e(j,3)) + 1, 2);
                        Errors(e(j,2), e(j,3)+n) = ...
                            mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                    end
                end
                % Introduce errors in the case of CNOT gate for
                % control and target qubits
                % Errors for control qubit are entry mod(e(j,1),4) 
                % according to standard indexing above
                if ( C(e(j,2),t) > 1000 )
                    if C(C(e(j,2),t) - 1000, t) == 1000
                        if mod(e(j,1),2) == 1
                            Errors(e(j,2), e(j,3)) = ...
                                mod(Errors(e(j,2), e(j,3)) + 1, 2);
                        end
                        if mod(e(j,1),4) > 1
                            Errors(e(j,2), e(j,3)+n) = ...
                                mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                        end
                        if mod(floor(e(j,1)/4),2) == 1
                            Errors(C(e(j,2),t) - 1000, e(j,3)) = ...
                                mod(Errors(C(e(j,2),t)-1000,e(j,3))+1,2);
                        end
                        if mod(floor(e(j,1)/4),4) > 1
                            Errors(C(e(j,2),t) - 1000, e(j,3)+n) = ...
                                mod(Errors(C(e(j,2),t)-1000,e(j,3)+n)+1,2);
                        end
                    end
                end
                % Introduce errors in the case of |0> prep
                if ( C(e(j,2),t) == 4 )
                    eVec = zeros(1,n);
                    if mod(e(j,1),2) == 1
                        % Need to translate the entries in the Prep
                        % tables to right format
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 2, a);
                        end
                        Errors(e(j,2), 1:n ) = ...
                            mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(e(j,1),4) > 1
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 3, a);
                        end
                        Errors(e(j,2), (n+1):end ) = ...
                            mod(Errors(e(j,2), (n+1):end)+eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),2) == 1
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 4, a);
                        end
                        Errors(e(j,2), 1:n ) = ...
                            mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),4) > 1
                        for a = 1:n
                            eVec(a) = ZPrepTable(e(j,3), 5, a);
                        end
                        Errors(e(j,2), (n+1):end ) = ...
                            mod(Errors(e(j,2), (n+1):end) + eVec, 2);
                    end
                end
                % Introduce errors in the case of |+> prep
                if ( C(e(j,2),t) == 3 )
                    eVec = zeros(1,n);
                    if mod(e(j,1),2) == 1
                        % Need to translate the entries in the Prep
                        % tables to right format
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 2, a);
                        end
                        Errors(e(j,2), 1:n ) = ...
                            mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(e(j,1),4) > 1
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 3, a);
                        end
                        Errors(e(j,2), (n+1):end ) = ...
                            mod(Errors(e(j,2), (n+1):end)+eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),2) == 1
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 4, a);
                        end
                        Errors(e(j,2), 1:n ) = ...
                            mod(Errors(e(j,2), 1:n) + eVec, 2);
                    end
                    if mod(floor(e(j,1)/4),4) > 1
                        for a = 1:n
                            eVec(a) = XPrepTable(e(j,3), 5, a);
                        end
                        Errors(e(j,2), (n+1):end ) = ...
                            mod(Errors(e(j,2), (n+1):end) + eVec, 2);
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

function result = Syndrome(errIn)
    global g
    result = mod((g * errIn')', 2);
end

function result = CheckLogicalSteane(err)
% returns 0 if err is a correctable error and 1 otherwise.

    % Compute the correction of the error
    CorrectionFromError(err);

    % Outputs 0 if err + corr is a stabilizer or 1 if it is a logical fault
    result = mod(sum(mod(err + corr, 2)), 2);

end

function result = CorrectionFromSyn(syn)
% returns a correction based on the measured syndrome using lookup table

    MatRecovery =  [0,0,0,0,0,0,0
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
	result = MatRecovery(correctionRow,:);

end

function result = CorrectionFromError(err)
% Compute the correction from error

    syn = Syndrome(err);
    result = CorrectionFromSyn(syn);
    
end

function result = ComputeLogicalUsingLookupTableX( ...
                  SynX1, SynX2, SynX3, SynX4, ErrX3, ErrX4)
% returns 1 if there is a logical X fault on 3rd or 4th blocks.
    
    % Store error recovery's based on the measured syndromes
    cxRow18 = CorrectionFromSyn(SynX1);
    cxRow19 = CorrectionFromSyn(SynX2);
    cxRow34 = CorrectionFromSyn(SynX3);
    cxRow35 = CorrectionFromSyn(SynX4);

    % Compute the final recovery using syndromes of all four blocks
    E1Final = mod(ErrX3 + cxRow18 + ...
        CorrectionFromError(mod(cxRow18 + cxRow34,2)),2);
    E3Final = mod(ErrX4 + cxRow18 + cxRow19 + ...
        CorrectionFromError(mod(cxRow18 + cxRow19 + cxRow35,2)),2);

    % Check if there is a logical X fault on the last two blocks
    ef1Full = mod(sum(mod(E1Final + CorrectionFromError(E1Final),2)),2);
    ef3Full = mod(sum(mod(E3Final + CorrectionFromError(E3Final),2)),2);

    result = 0;
    if (ef1Full == 1) || (ef3Full == 1)
        result = 1;
    end

end
