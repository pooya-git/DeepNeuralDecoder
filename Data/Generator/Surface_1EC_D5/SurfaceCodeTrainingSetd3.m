% MIT License
%
% Copyright (c) 2018 Chris Chamberland
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function SurfaceCodeTrainingSetd5
n=1;
t=2;

parfor i = 1:6
    
    numIterations = 2*10^2;
    v = [3*10^-4,4*10^-4,5*10^-4,6*10^-4,7*10^-4,8*10^-4];    
    errRate = v(1,i);


   str_errRate = num2str(errRate,'%0.3e');       
    
    [OutputSynX1,OutputSynZ1,OutputErrX1,OutputErrZ1,OutputCount] = depolarizingSimulator(numIterations,errRate,n,t);
    A1 = [OutputSynX1,OutputSynZ1];
    A2 = [OutputErrX1,OutputErrZ1];
    
    %clear OutputSynX1 OutputSynZ1 OutputErrX1 OutputErrZ1
    
    TempStr1 = 'SyndromeOnly';
    TempStr2 = '.txt';
    TempStr3 = 'ErrorOnly';   
    str_Final1 = strcat(TempStr1,str_errRate);
    str_Final2 = strcat(str_Final1,TempStr2);
    str_Final3 = strcat(TempStr3,str_errRate);
    str_Final4 = strcat(str_Final3,TempStr2);
    fid = fopen(str_Final2, 'w+t');
    for ii = 1:size(A1,1)
        fprintf(fid,'%g\t',A1(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
    fid = fopen(str_Final4, 'w+t');
    for ii = 1:size(A2,1)
        fprintf(fid,'%g\t',A2(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    %clear A1 A2
    
    str_Count = strcat('Count',str_errRate);
    str_CountFinal = strcat(str_Count,'.mat');
    parsaveCount(str_CountFinal,OutputCount);
    
end


end


function parsaveErrorVec(fname,errorVecMat)
save(fname,'errorVecMat');
end

function parsaveCount(fname,OutputCount)
save(fname,'OutputCount');
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
            
            if (C(i,t) > 1000) && (C(i,t) < 2000)
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
            
            if C(i,t) > 2000
                % check to see if target qubit according to control qubit is actually a target qubit
                if C(C(i,t) - 2000, t) == 2000
                    v1 = Errors(i, 1:n);
                    v2 = Errors(i, (n+1):end);
                    w1 = Errors(C(i,t) - 2000, 1:n);
                    w2 = Errors(C(i,t) - 2000, (n+1):end);
         
                    Errors(C(i,t) - 2000, 1:n) = v1;
                    Errors(C(i,t) - 2000, (n+1):end) = v2;
                    Errors(i, 1:end) = w1;
                    Errors(i, (n+1):end) = w2;
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
                if ( (C(e(j,2),t) > 1000) && (C(e(j,2),t) < 2000))
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
                if C(e(j,2),t) > 2000
                    if C(C(e(j,2),t) - 2000, t) == 2000
                        if mod(e(j,1),2) == 1
                            Errors(e(j,2), e(j,3)) = mod(Errors(e(j,2), e(j,3)) + 1, 2);
                        end
                        if mod(e(j,1),4) > 1
                            Errors(e(j,2), e(j,3)+n) = mod(Errors(e(j,2), e(j,3)+n) + 1, 2);
                        end
                        if mod(floor(e(j,1)/4),2) == 1
                            Errors(C(e(j,2),t) - 2000, e(j,3)) = mod(Errors(C(e(j,2),t) - 2000, e(j,3)) + 1, 2);
                        end
                        if mod(floor(e(j,1)/4),4) > 1
                            Errors(C(e(j,2),t) - 2000, e(j,3)+n) = mod(Errors(C(e(j,2),t) - 2000, e(j,3)+n) + 1, 2);
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

function Output = SurfaceCodeCircuitGenerator(d)

% Creates d = 3 surface code circuit Matteo

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
%2000: Qubit for SWAP gate

numRows = 2*(d^2) -1 ;
numTimeSteps = 6;
qubitNum = (d^2-1)/2;

Cd3 = zeros(numRows,numTimeSteps);

% Set the storage locations at all qubit locations (first two and last two
% time steps)
for i = 1:(d^2)
    for j = 1:numTimeSteps
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
    end
end


% Initialise X stabilizer state prep and measurements
for i = 1:((d^2)-1)/2
    Cd3(i,1) = 3;
    Cd3(i,6) = 5;
end

% Initialise Z stabilizer state prep and measurements
for i = (((3*(d^2)-1)/2)+1):(2*(d^2)-1)
    Cd3(i,1) = 4;
    Cd3(i,6) = 6;
end

% Creat matrix of data qubit number
dataMat = zeros(d,d);
counter = 1;
for i = 1:d
    for j = 1:d
        dataMat(i,j) = counter + (((d^2)-1)/2);
        counter = counter + 1;
    end
end

% Creat matrix for X and Z stabilizer ancilla qubit numbers
ancillaStabXMat = zeros(((d+1)/2),d-1);
ancillaStabZMat = zeros(d-1,((d+1)/2));

counter = 1;
for i = 1:length(ancillaStabXMat(1,:))
    for j = 1:length(ancillaStabXMat(:,1))
        ancillaStabXMat(j,i) = counter;
        counter = counter + 1;
    end
end

counter = 1;
for i = 1:length(ancillaStabZMat(:,1))
    for j = 1:length(ancillaStabZMat(1,:))
        ancillaStabZMat(i,j) = counter;
        counter = counter + 1;
    end
end

% Next we input gates from measurement qubits to data qubits 

% First cycle (measure upper right qubits)
% Input target and control of the CNOT gates for X stabilizers
timeStep = 2;
rwoXstab = 1;
for i = 1:length(dataMat(:,1))
    if mod(i,2) == 0
        colXstab = 2;
    else
        colXstab = 1;
    end
    for j = 1:length(dataMat(1,:))
        if mod(i,2) == 1
            if mod(j,2) == 0
                Cd3(dataMat(i,j),timeStep) = 1000;
                Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
                colXstab = colXstab + 2;
            end
        else
            if mod(j,2) == 1 && j ~= 1
                Cd3(dataMat(i,j),timeStep) = 1000;
                Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
                colXstab = colXstab + 2;
            end
        end
    end
    if mod(i,2) == 1
        rwoXstab = rwoXstab + 1;
    end
end

% Input target and control of the CNOT gates for Z stabilizers
rwoZstab = 1;
for i = 1:(length(dataMat(:,1))-1)
    colZstab = 1;
    for j = 1:length(dataMat(1,:))
        if mod(i,2) == 1
            if mod(j,2) == 1
                Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
                Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
                colZstab = colZstab + 1;
            end
        else
            if mod(j,2) == 0
                Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
                Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
                colZstab = colZstab + 1;
            end
        end
    end
    rwoZstab = rwoZstab + 1;
end

% Second cycle (measure upper left qubits for X stabilizers and lower right qubits for Z stabilizers)
timeStep = timeStep + 1;

% Input target and control of the CNOT gates for X stabilizers
rwoXstab = 1;
for i = 1:length(dataMat(:,1))
    if mod(i,2) == 0
        colXstab = 2;
    else
        colXstab = 1;
    end
    for j = 1:length(dataMat(1,:))
        if (mod(i,2) == 1) && (mod(j,2) == 1) && (j < length(dataMat(1,:)))
            Cd3(dataMat(i,j),timeStep) = 1000;
            Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
            colXstab = colXstab + 2;
        elseif (mod(i,2) == 0) && (mod(j,2) == 0)
            Cd3(dataMat(i,j),timeStep) = 1000;
            Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
            colXstab = colXstab + 2;
        end
    end
    if mod(i,2) == 1
        rwoXstab = rwoXstab + 1;
    end
end

% Input target and control of the CNOT gates for Z stabilizers
rwoZstab = 1;
for i = 2:length(dataMat(:,1))
    colZstab = 1;
    for j = 1:length(dataMat(1,:))
        if (mod(i,2) == 0) && (mod(j,2) == 1)
            Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
            Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
            colZstab = colZstab + 1;
        elseif (mod(i,2) == 1) && (mod(j,2) == 0)
            Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
            Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
            colZstab = colZstab + 1;
        end
    end
    rwoZstab = rwoZstab + 1;
end

% Third cycle (lower right qubits for X stabilizers and upper left qubits for Z stabilizers)
timeStep = timeStep + 1;

% Input target and control of the CNOT gates for X stabilizers
rwoXstab = 1;
for i = 1:length(dataMat(:,1))
    if mod(i,2) == 0
        colXstab = 1;
    else
        colXstab = 2;
    end
    for j = 2:length(dataMat(1,:))
        if mod(i,2) == 1 && mod(j,2) == 1
            Cd3(dataMat(i,j),timeStep) = 1000;
            Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
            colXstab = colXstab + 2;
        elseif mod(i,2) == 0 && mod(j,2) == 0
            Cd3(dataMat(i,j),timeStep) = 1000;
            Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
            colXstab = colXstab + 2;
        end
    end
    if mod(i,2) == 0
        rwoXstab = rwoXstab + 1;
    end
end

% Input target and control of the CNOT gates for Z stabilizers
rwoZstab = 1;
for i = 1:(length(dataMat(:,1))-1)
    if mod(i,2) == 1
        colZstab = 2;
    else
        colZstab = 1;
    end
    for j = 1:length(dataMat(1,:))
        if mod(i,2) == 1 && mod(j,2) == 0
            Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
            Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
            colZstab = colZstab + 1;
        elseif mod(i,2) == 0 && mod(j,2) == 1
            Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
            Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
            colZstab = colZstab + 1;
        end
    end
    rwoZstab = rwoZstab + 1;
end

% Fourth cycle (lower right qubits for X stabilizers and upper left qubits for Z stabilizers)
timeStep = timeStep + 1;

% Input target and control of the CNOT gates for X stabilizers
rwoXstab = 1;
for i = 1:length(dataMat(:,1))
    if mod(i,2) == 0
        colXstab = 1;
    else
        colXstab = 2;
    end
    for j = 1:(length(dataMat(1,:))-1)
        if mod(i,2) == 1 && mod(j,2) == 0
            Cd3(dataMat(i,j),timeStep) = 1000;
            Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
            colXstab = colXstab + 2;
        elseif mod(i,2) == 0 && mod(j,2) == 1
            Cd3(dataMat(i,j),timeStep) = 1000;
            Cd3(ancillaStabXMat(rwoXstab,colXstab),timeStep) = 1000 + dataMat(i,j);
            colXstab = colXstab + 2;
        end
    end
    if mod(i,2) == 0
        rwoXstab = rwoXstab + 1;
    end
end

% Input target and control of the CNOT gates for Z stabilizers
rwoZstab = 1;
for i = 2:length(dataMat(:,1))
    if mod(i,2) == 0
        colZstab = 2;
    else
        colZstab = 1;
    end
    for j = 1:length(dataMat(1,:))
        if mod(i,2) == 0 && mod(j,2) == 0
            Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
            Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
            colZstab = colZstab + 1;
        elseif mod(i,2) == 1 && mod(j,2) == 1
            Cd3(ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2,timeStep) = 1000;
            Cd3(dataMat(i,j),timeStep) = 1000 + ancillaStabZMat(rwoZstab,colZstab)+d^2+((d^2)-1)/2;
            colZstab = colZstab + 1;
        end
    end
    rwoZstab = rwoZstab + 1;
end

Output = Cd3;

end

function Output = ErrorGenerator(Cmat,errRate)
%This function outputs an error vector based on the input circuit
%represented by Cmat. We have the following noise model.

% 1) |0> state preparation: Perfect |0> state followed by X error with probability p
% 2) Z-measurement: X pauli with probability p followed by perfect Z-basis
% measurement.
% 3) CNOT: Perfect CNOT followed by
%{IX,IY,IZ,XI,YI,ZI,XX,XY,XZ,ZX,ZY,ZZ,YX,YY,YZ} with probability p/15 each.
% 4) Hadamard: Perfect Hadamard followed by {X,Y,Z} with probability p/12
%each.
% 5) SWAP: Perfect SWAP followed by
% {IX,IY,IZ,XI,YI,ZI,XX,XY,XZ,ZX,ZY,ZZ,YX,YY,YZ} with probability p/60
% each.
% Storage: Pauli {X,Y,Z} with probability p/30 each.

% Here errRate = p and Cmat is the circuit representing the surface code
% lattice.


e = zeros(1,4);
counter = 1;

for i = 1:length(Cmat(:,1))
    for j = 1:length(Cmat(1,:))
        
        % Adds storage errors with probability p
        if (Cmat(i,j) == 1)
            xi = rand;
            if xi < errRate
                k = randi([1,3]);
                e(counter,:) = [k,i,1,j];
                counter = counter + 1;
            end
        end
        
        % Adds state-preparation errors with probability 2p/3
        if Cmat(i,j) == 4
            xi = rand;
            if xi < 2*errRate/3
                e(counter,:) = [1,i,1,j];
                counter = counter + 1;
            end
        end
        if Cmat(i,j) == 3
            xi = rand;
            if xi < 2*errRate/3
                e(counter,:) = [2,i,1,j];
                counter = counter + 1;
            end
        end
        
        % Adds measurement errors with probability 2p/3
        if Cmat(i,j) == 6
            xi = rand;
            if xi < 2*errRate/3
                e(counter,:) = [1,i,1,j];
                counter = counter + 1;
            end
        end
        if Cmat(i,j) == 5
            xi = rand;
            if xi < 2*errRate/3
                e(counter,:) = [2,i,1,j];
                counter = counter + 1;
            end
        end
        
        
        % Adds CNOT errors with probability p
        if (Cmat(i,j) > 1000)
            xi = rand;
            if xi < errRate
                k = randi([1,15]);
                e(counter,:) = [k,i,1,j];
                counter = counter + 1;
            end
        end
        
    end
end

Output = e;

end

function Output = ConvertErrorXStringToErrorVec(ErrStrX)

e = zeros(1,4);
counter = 1;

for i = 1:length(ErrStrX(1,:))
    if ErrStrX(1,i) == 1
        e(counter,:) = [1,i+12,1,1];
        counter = counter + 1;
    end
end

Output = e;

end

function Output = ConvertErrorZStringToErrorVec(ErrStrZ)

e = zeros(1,4);
counter = 1;

for i = 1:length(ErrStrZ(1,:))
    if ErrStrZ(1,i) == 1
        e(counter,:) = [2,i+12,1,1];
        counter = counter + 1;
    end
end

Output = e;

end

function [OutputSynX1,OutputSynZ1,OutputErrX1,OutputErrZ1,OutputCount] = depolarizingSimulator(numIterations,errRate,n,t)

% This function generates X and Z syndrome measurement results for three 
% rounds of error correction of the d=3 rotated surface code as well as the
% X and Z errors for each round

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

% Full circuit for measuring the X and Z stabilizers of the d=3 rotated 
% surface code 
Circuit = [3        1014        1013        1019        1018           5
           3        1024        1023        1029        1028           5
           3        1034        1033           0           0           5
           3           0           0        1015        1014           5
           3        1020        1019        1025        1024           5
           3        1030        1029        1035        1034           5
           3        1016        1015        1021        1020           5
           3        1026        1025        1031        1030           5
           3        1036        1035           0           0           5
           3           0           0        1017        1016           5
           3        1022        1021        1027        1026           5
           3        1032        1031        1037        1036           5
           1        1038        1000           1           1           1
           1        1000           1        1039        1000           1
           1        1039        1000        1000           1           1
           1        1000           1        1040        1000           1
           1        1040           1        1000           1           1
           1           1        1038        1041        1000           1
           1        1041        1000        1000        1039           1
           1        1000        1039        1042        1000           1
           1        1042        1000        1000        1040           1
           1        1000        1040        1043           1           1
           1        1044        1000           1        1041           1
           1        1000        1041        1045        1000           1
           1        1045        1000        1000        1042           1
           1        1000        1042        1046        1000           1
           1        1046           1        1000        1043           1
           1           1        1044        1047        1000           1
           1        1047        1000        1000        1045           1
           1        1000        1045        1048        1000           1
           1        1048        1000        1000        1046           1
           1        1000        1046        1049           1           1
           1           1        1000           1        1047           1
           1        1000        1047           1        1000           1
           1           1        1000        1000        1048           1
           1        1000        1048           1        1000           1
           1           1           1        1000        1049           1
           4        1000        1000           0           0           6
           4        1000        1000        1000        1000           6
           4        1000        1000        1000        1000           6
           4        1000        1000        1000        1000           6
           4        1000        1000        1000        1000           6
           4           0           0        1000        1000           6
           4        1000        1000           0           0           6
           4        1000        1000        1000        1000           6
           4        1000        1000        1000        1000           6
           4        1000        1000        1000        1000           6
           4        1000        1000        1000        1000           6
           4           0           0        1000        1000           6];
 
totalIt = 0.5*((t^2)+3*t+2);
ErrXOutputBlock1 = zeros(totalIt*numIterations,25);
ErrZOutputBlock1 = zeros(totalIt*numIterations,25);

SynXOutputBlock1 = zeros(totalIt*numIterations,12);
SynZOutputBlock1 = zeros(totalIt*numIterations,12);

numSize = 1;
countIterations = 0;
while numSize < (numIterations+1)
      
    XSyn = zeros(totalIt,12); % Stores the X syndromes for each round
    ZSyn = zeros(totalIt,12); % Stores the Z syndromes for each round
    XErrTrack = zeros(totalIt,25); % Stores the X errors for each round
    ZErrTrack = zeros(totalIt,25); % Stores the Z errors for each round
    e = zeros(1,4);      
    
    for numRound = 1:6
        eCircuit = ErrorGenerator(Circuit,errRate);
        eTemp = [e;eCircuit];
        Cout = transpose(PropagationStatePrepArb(Circuit, n, eTemp));
        Xerror = Cout(1,1:2:50);
        Zerror = Cout(1,2:2:50);
        XErrTrack(numRound,:) = Xerror;
        ZErrTrack(numRound,:) = Zerror;
        ZSyn(numRound,:) = Cout(1,51:62);
        XSyn(numRound,:) = Cout(1,63:74);
        eX = ConvertErrorXStringToErrorVec(Xerror);
        eZ = ConvertErrorZStringToErrorVec(Zerror);
        e = [eX;eZ];       
    end
       
    
    if sum(any(XSyn)) ~= 0 || sum(any(ZSyn)) ~= 0 || sum(any(Xerror)) ~= 0 || sum(any(Zerror)) ~= 0
        ErrXOutputBlock1(totalIt*(numSize-1)+1:totalIt*numSize,:) = XErrTrack;
        ErrZOutputBlock1(totalIt*(numSize-1)+1:totalIt*numSize,:) = ZErrTrack;
        SynXOutputBlock1(totalIt*(numSize-1)+1:totalIt*numSize,:) = XSyn;
        SynZOutputBlock1(totalIt*(numSize-1)+1:totalIt*numSize,:) = ZSyn;
        numSize = numSize + 1;
    end
    
    countIterations = countIterations + 1;
    
end

OutputSynX1 = SynXOutputBlock1;
OutputSynZ1 = SynZOutputBlock1;
OutputErrX1 = ErrXOutputBlock1;
OutputErrZ1 = ErrZOutputBlock1;
OutputCount = countIterations;

end