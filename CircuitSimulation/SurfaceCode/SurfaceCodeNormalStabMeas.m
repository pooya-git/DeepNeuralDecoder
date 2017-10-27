function SurfaceCodeNormalStabMeas
n=1;
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

% Testing the function call
% errRate = 10^(-3);
% d = 5;
% noiseSteps = 3;
% errorModel =3;
% numIterations = 10000;
%
% [errRateX,errRateZ,errRateY] = simulationSurfaceCode(errRate,d,noiseSteps,errorModel,numIterations)
% %%%%%
%
% return


numIterations = 10^6;
d = 3;
errorModel = 2;
noiseSteps = d;


parfor NumNoisemodel = 1:19
    v = [10^-4,2*10^-4,3*10^-4,4*10^-4,5*10^-4,6*10^-4,7*10^-4,8*10^-4,9*10^-4,10^-3,2*10^-3,3*10^-3,4*10^-3,5*10^-3,6*10^-3,7*10^-3,8*10^-3,9*10^-3,10^-2];
    errRate = v(1,NumNoisemodel);
    %errRate = 10^-2;
    
    MatXFinal = zeros(numIterations*noiseSteps*(d+1)/2,d-1);
    MatZFinal = zeros(numIterations*noiseSteps*(d-1),(d+1)/2);
    XerrFinal = zeros(numIterations*noiseSteps,d^2);
    ZerrFinal = zeros(numIterations*noiseSteps,d^2);
    
    for ii = 1:numIterations
        
        
        %     str_errRate = num2str(errRate,'%0.3e');
        %     errVecStringXnoLeak = 'errVecXNoLeakage';
        %     errVecStringYnoLeak = 'errVecYNoLeakage';
        %     errVecStringZnoLeak = 'errVecZNoLeakage';
        %     errVecStringTotalnoLeak = 'errVecTotalNoLeakage';
        %     str_errVecFulltempXnoLeak = strcat(errVecStringXnoLeak,str_errRate);
        %     str_errVecFulltempYnoLeak = strcat(errVecStringYnoLeak,str_errRate);
        %     str_errVecFulltempZnoLeak = strcat(errVecStringZnoLeak,str_errRate);
        %     str_errVecFulltempTotoalnoLeak = strcat(errVecStringTotalnoLeak,str_errRate);
        %     str_errVecFullXnoLeak = strcat(str_errVecFulltempXnoLeak,'.mat');
        %     str_errVecFullYnoLeak = strcat(str_errVecFulltempYnoLeak,'.mat');
        %     str_errVecFullZnoLeak = strcat(str_errVecFulltempZnoLeak,'.mat');
        %     str_errVecFullTotalnoLeak = strcat(str_errVecFulltempTotoalnoLeak,'.mat');
        %
        %     % Tomas this where your function should be called. Output = [logX/numIterations,logZ/numIterations,logY/numIterations]
        %     [MXGrid,MZGrid] = MxMzGrid(errRate,d,noiseSteps,errorModel);
        % %     parsaveErrorVec(str_errVecFullXnoLeak,errRateX);
        % %     parsaveErrorVec(str_errVecFullYnoLeak,errRateY);
        % %     parsaveErrorVec(str_errVecFullZnoLeak,errRateZ);
        %     parsaveErrorVec(str_errVecFullTotalnoLeak,errRateTotal);
        [MeasXMatOut,MeasZMatOut,XErrMatOut,ZErrMatOut] = MxMzGrid(errRate,d,noiseSteps,errorModel);
        
        MatXFinal(((ii-1)*noiseSteps*(d+1)/2)+1:ii*noiseSteps*(d+1)/2,:) = MeasXMatOut;
        MatZFinal(((ii-1)*noiseSteps*(d-1))+1:ii*noiseSteps*(d-1),:) = MeasZMatOut;
        XerrFinal(((ii-1)*noiseSteps)+1:ii*noiseSteps,:) = XErrMatOut;
        ZerrFinal(((ii-1)*noiseSteps)+1:ii*noiseSteps,:) = ZErrMatOut;
        
        
    end
    
    str_errRate = num2str(errRate,'%0.3e');
    TempStr1X = 'XSyndrome';
    TempStr1Z = 'ZSyndrome';
    ErrStrX = 'Xerror';
    ErrStrZ = 'Xerror';
    TempStr2 = '.txt';
    
    str_Final1Xsyn = strcat(TempStr1X,str_errRate);
    str_Final2Xsyn = strcat(str_Final1Xsyn,TempStr2);
    
    str_Final1Zsyn = strcat(TempStr1Z,str_errRate);
    str_Final2Zsyn = strcat(str_Final1Zsyn,TempStr2);
    
    str_Final1Xerr = strcat(ErrStrX,str_errRate);
    str_Final2Xerr = strcat(str_Final1Xerr,TempStr2);
    
    str_Final1Zerr = strcat(ErrStrZ,str_errRate);
    str_Final2Zerr = strcat(str_Final1Zerr,TempStr2);
    
    fid = fopen(str_Final2Xsyn, 'w+t' );
    for ii = 1:size(MatXFinal,1)
        fprintf(fid,'%g\t',MatXFinal(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    fid = fopen(str_Final2Zsyn, 'w+t' );
    for ii = 1:size(MatZFinal,1)
        fprintf(fid,'%g\t',MatZFinal(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    fid = fopen(str_Final2Xerr, 'w+t' );
    for ii = 1:size(XerrFinal,1)
        fprintf(fid,'%g\t',XerrFinal(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    fid = fopen(str_Final2Zerr, 'w+t' );
    for ii = 1:size(ZerrFinal,1)
        fprintf(fid,'%g\t',ZerrFinal(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
    
end


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

function Output = vecX(vecerr,d)
% Outputs a vector containing all the X errors. The i'th column is 0 if no
% X error on qubit i and 1 otherwise.

vecerrNew = vecerr(1:2*d^2,1);

len = length(vecerrNew)/2;

vecX = zeros(1,len);

counter = 1;
for i = 1:length(vecerrNew)
    if mod(i,2) == 1
        vecX(1,counter) =  vecerrNew(i);
        counter = counter + 1;
    end
end

Output = vecX;

end

function Output = vecZ(vecerr,d)
% Outputs a vector containing all the Z errors. The i'th column is 0 if no
% Z error on qubit i and 1 otherwise.

vecerrNew = vecerr(1:2*d^2,1);

len = length(vecerrNew)/2;

vecX = zeros(1,len);

counter = 1;
for i = 1:length(vecerrNew)
    if mod(i,2) == 0
        vecX(1,counter) =  vecerrNew(i);
        counter = counter + 1;
    end
end

Output = vecX;
end

function Output = MeasureZmatMatteo(vec,d)
% Outputs the measurement Z matrix. A 1 indicates a measurement that
% flagged. The i'th row and j'th column corresponds to measurement Z qubit
% at the i'th row  and j'th column of the measure Z qubit lattice.

len = length(vec);

vecNew = vec((2*d^2)+1:len,1);

MeasZ = zeros(d-1,(d+1)/2);

counter = 1;
for i = 1:(d-1)
    if mod(i,2) == 1
        MeasZ(i,1) = vecNew(counter);
        counter = counter + 1;
    end
end

for i = 1:(d-1)
    for j = 1:((d+1)/2)
        if (mod(i,2) == 1) && ( j ~= 1)
            MeasZ(i,j) = vecNew(counter);
            counter = counter + 1;
        elseif (mod(i,2) == 0) && ( j ~= (d+1)/2)
            MeasZ(i,j) = vecNew(counter);
            counter = counter + 1;
        end
    end
end

for i = 1:(d-1)
    if mod(i,2) == 0
        MeasZ(i,(d+1)/2) = vecNew(counter);
        counter = counter + 1;
    end
end

Output = MeasZ;
end

function Output = MeasureXmatMatteo(vec,d)
% Outputs the measurement Z matrix. A 1 indicates a measurement that
% flagged. The i'th row and j'th column corresponds to measurement Z qubit
% at the i'th row  and j'th column of the measure Z qubit lattice.

len = length(vec);

vecNew = vec((2*d^2)+1:len,1);

MeasX = zeros((d+1)/2,d-1);

counter = 0.5*((d-1)^2) + d;

for i = 1:((d+1)/2)
    for j =1:(d-1)
        if mod(j,2) == 0
            MeasX(i,j) = vecNew(counter);
            counter = counter + 1;
        end
    end
    
    for j =1:(d-1)
        if mod(j,2) == 1
            MeasX(i,j) = vecNew(counter);
            counter = counter + 1;
        end
    end
    
end

Output = MeasX;

end

function Output = errorMatConverter(xErr,zErr,d)
% Converts xErr and zErr bit strings into error matrix

qubitNum = 0.5*(d-1)*(5*d+1);

% Convert xErr and zErr to error matrix
e = zeros(1,4);
counter = 1;
for ii = 1:length(xErr)
    if xErr(1,ii) == 1
        e(counter,:) = [1,ii+qubitNum,1,1];
        counter = counter + 1;
    end
end

for ii = 1:length(zErr)
    if zErr(1,ii) == 1
        e(counter,:) = [2,ii+qubitNum,1,1];
        counter = counter + 1;
    end
end

Output = e;

end

function Output = circuitSurfaced3Matteov2(d)

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

numRows = 6*(d^2) - 4*d - 1;
numTimeSteps = 16;
qubitNum = 0.5*(d-1)*(5*d+1);

Cd3 = zeros(numRows,numTimeSteps);

% Set the storage locations at all qubit locations (first two and last two
% time steps)
for i = 1:(d^2)
    for j = [1,numTimeSteps]
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
    end
end


for i = 1:(d^2)    
    for j = [1,4,7,10,13]
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
        Cd3(i + qubitNum,j) = 1;
    end
end

% Deactiviate ancilla qubits in last two time steps and initialize
% measurement + ancilla qubits

% First round of weight-two X stabilizers
listX2cycle1 = zeros(1,(d-1)/2);
counter = 1;
for i = 1:length(listX2cycle1(1,:))
    listX2cycle1(1,i) = counter;
    counter = counter + 3;
end

for i = 1:3*(d-1)/2
    if any(i==listX2cycle1)
        Cd3(i,1) = 4;
        Cd3(i,2) = 10;
        Cd3(i,15) = 10;
        Cd3(i,16) = 6;
    else
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = -1;
        Cd3(i,16) = -1;
    end
end

% Weight-four X stabilizers
listX4 = zeros(1,((d-1)^2)/2);
counter = 3*(d-1)/2 + 1;
for i = 1:length(listX4)
    listX4(1,i) = counter;
    counter = counter + 5;
end

for i = ((3*(d-1)/2) + 1):((3*(d-1)/2) + (5*((d-1)^2)/2))
    if any(i==listX4)
        Cd3(i,1) = 4;
        Cd3(i,2) = 10;
        Cd3(i,15) = 10;
        Cd3(i,16) = 6;
    else
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = -1;
        Cd3(i,16) = -1;
    end
end

% Second round of weight-two X stabilizers
listX2cycle2 = zeros(1,(d-1)/2);
counter = (3*(d-1)/2) + (5*((d-1)^2)/2) + 1;
for i = 1:length(listX2cycle2(1,:))
    listX2cycle2(1,i) = counter;
    counter = counter + 3;
end

for i = ((3*(d-1)/2) + (5*((d-1)^2)/2) + 1):0.5*(d-1)*(5*d+1)
    if any(i==listX2cycle2)
        Cd3(i,1) = 4;
        Cd3(i,2) = 10;
        Cd3(i,15) = 10;
        Cd3(i,16) = 6;
    else
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = -1;
        Cd3(i,16) = -1;
    end
end

% First round of weight-two Z stabilizers
listZ2cycle1 = zeros(1,(d-1)/2);
counter = 0.5*(d-1)*(5*d+1) + d^2 + 3;
for i = 1:length(listZ2cycle1(1,:))
    listZ2cycle1(1,i) = counter;
    counter = counter + 3;
end

for i = (0.5*(d-1)*(5*d+1) + d^2 + 1):(0.5*(d-1)*(5*d+1) + d^2 + (3*(d-1)/2))
    if any(i==listZ2cycle1)
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = 6;
        Cd3(i,16) = -1;
    else
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = -1;
        Cd3(i,16) = -1;
    end
end

% Weight-four Z stabilizers
listZ4 = zeros(1,((d-1)^2)/2);
counter = 0.5*(d-1)*(5*d+1) + d^2 + (3*(d-1)/2) + 5;
for i = 1:length(listZ4)
    listZ4(1,i) = counter;
    counter = counter + 5;
end

for i = (0.5*(d-1)*(5*d+1) + d^2 + (3*(d-1)/2) + 1):(0.5*(d-1)*(5*d+1) + d^2 + (3*(d-1)/2) + 5*(((d-1)^2)/2))
    if any(i==listZ4)
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = 6;
        Cd3(i,16) = -1;
    else
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = -1;
        Cd3(i,16) = -1;
    end
end


% Second round of weight-two Z stabilizers
listZ2cycle2 = zeros(1,(d-1)/2);
counter = 0.5*(d-1)*(5*d+1) + d^2 + (3*(d-1)/2) + 5*(((d-1)^2)/2) + 3;
for i = 1:length(listZ2cycle2(1,:))
    listZ2cycle2(1,i) = counter;
    counter = counter + 3;
end

for i = (0.5*(d-1)*(5*d+1) + d^2 + (3*(d-1)/2) + 5*(((d-1)^2)/2) + 1):numRows
    if any(i==listZ2cycle2)
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = 6;
        Cd3(i,16) = -1;
    else
        Cd3(i,1) = 0;
        Cd3(i,2) = 4;
        Cd3(i,15) = -1;
        Cd3(i,16) = -1;
    end
end

% Creat matrix of data qubit number
dataMat = zeros(d,d);
counter = 1;
for i = 1:d
    for j = 1:d
        dataMat(i,j) = counter + 0.5*(d-1)*(5*d+1);
        counter = counter + 1;
    end
end

% Next we input gates from measurement qubits to data qubits (and ancilla's
% for the SWAP gates)

% listX2cycle1, listX4, listX2cycle2;
% listZ2cycle1, listZ4, listZ2cycle2;



% First cycle (measure upper left qubits)

% Generate list of target qubits for listX4
listTarX4UpperLeft = zeros(1,length(listX4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1) && ((mod(j,2) == 1) && j ~= d)
            listTarX4UpperLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && ((mod(j,2) == 0) && j ~= d)
            listTarX4UpperLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listLowestRowUpperLeft = listTarX4UpperLeft(1,(((d-1)^2)/2)+1:length(listTarX4UpperLeft(1,:))); % list of target qubits at the last row of the lattice (where the target qubit is at position upper left)

% Inserts gates for measuring X of the upper left target qubits
for i = 1:length(listX4(1,:))
    Cd3 = XmeasurementCircuitv2Mid(Cd3,listTarX4UpperLeft(i),listX4(i),2);
end

for i = 1:length(listLowestRowUpperLeft(1,:))
    Cd3 = XmeasurementCircuitv2UpLow(Cd3,listLowestRowUpperLeft(i),listX2cycle2(i),2);
end

% Generate list of target qubits for listZ4
listTarZ4UpperLeft = zeros(1,length(listZ4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1 && i ~= d ) && (mod(j,2) == 0)
            listTarZ4UpperLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && ((mod(j,2) == 1) && j ~= d)
            listTarZ4UpperLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listLastCol = zeros(1,length(listZ2cycle2(1,:)));
counter = 1;
for i = 1:d
    if mod(i,2) == 0 
        listLastCol(1,counter) = dataMat(i,d);
        counter = counter + 1;
    end
end

% Inserts gates for measuring Z of the upper left target qubits
for i = 1:length(listZ4(1,:))
    Cd3 = ZmeasurementCircuitv2Mid(Cd3,listTarZ4UpperLeft(i),listZ4(i),3);
end

for i = 1:length(listLastCol(1,:))
    Cd3 = ZmeasurementCircuitv2LeftRight(Cd3,listLastCol(i),listZ2cycle2(i),3);
end







% Second cycle (measure upper right qubits)

listTarX4UpperRight = zeros(1,length(listX4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1) && (mod(j,2) == 0)
            listTarX4UpperRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && ((mod(j,2) == 1) && j > 1)
            listTarX4UpperRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listLowestRowUpperRight = listTarX4UpperRight(1,(((d-1)^2)/2)+1:length(listTarX4UpperRight(1,:))); % list of target qubits at the last row of the lattice (where the target qubit is at position upper left)

% Inserts gates for measuring X of the upper right target qubits
for i = 1:length(listX4(1,:))
    Cd3 = XmeasurementCircuitv2Mid(Cd3,listTarX4UpperRight(i),listX4(i),1);
end

for i = 1:length(listLowestRowUpperLeft(1,:))
    Cd3 = XmeasurementCircuitv2UpLow(Cd3,listLowestRowUpperRight(i),listX2cycle2(i),1);
end



% Generate list of target qubits for listZ4
listTarZ4UpperRight = zeros(1,length(listZ4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1 && (i ~= d)) && (mod(j,2) == 1 && (j > 1))
            listTarZ4UpperRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && (mod(j,2) == 0)
            listTarZ4UpperRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listFirstCol = zeros(1,length(listZ2cycle1(1,:)));
counter = 1;
for i = 1:d
    if (mod(i,2) == 1) && (i ~= d) 
        listFirstCol(1,counter) = dataMat(i,1);
        counter = counter + 1;
    end
end

% Inserts gates for measuring Z of the upper left target qubits
for i = 1:length(listZ4(1,:))
    Cd3 = ZmeasurementCircuitv2Mid(Cd3,listTarZ4UpperRight(i),listZ4(i),1);
end

for i = 1:length(listFirstCol(1,:))
    Cd3 = ZmeasurementCircuitv2LeftRight(Cd3,listFirstCol(i),listZ2cycle1(i),1);
end




% Third cycle (measure lower right qubits)

listTarX4LowerRight = zeros(1,length(listX4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1) && ((mod(j,2) == 1) && j > 1)
            listTarX4LowerRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && (mod(j,2) == 0)
            listTarX4LowerRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listHighestRowLowerRight = listTarX4LowerRight(1,1:length(listX2cycle1(1,:)));

listTarX4LowerRight = listTarX4LowerRight(1,length(listHighestRowLowerRight(1,:))+1:length(listTarX4LowerRight(1,:)));


% Inserts gates for measuring X of the lower right target qubits
for i = 1:length(listX4(1,:))
    Cd3 = XmeasurementCircuitv2Mid(Cd3,listTarX4LowerRight(i),listX4(i),3);
end

for i = 1:length(listLowestRowUpperLeft(1,:))
    Cd3 = XmeasurementCircuitv2UpLow(Cd3,listHighestRowLowerRight(i),listX2cycle1(i),3);
end

% Generate list of target qubits for listZ4
listTarZ4LowerRight = zeros(1,length(listZ4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1 && (i > 1)) && (mod(j,2) == 0)
            listTarZ4LowerRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && ((mod(j,2) == 1) && j > 1)
            listTarZ4LowerRight(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listFirstCol = zeros(1,length(listZ2cycle1(1,:)));
counter = 1;
for i = 1:d
    if (mod(i,2) == 0) 
        listFirstCol(1,counter) = dataMat(i,1);
        counter = counter + 1;
    end
end

% Inserts gates for measuring Z of the upper left target qubits
for i = 1:length(listZ4(1,:))
    Cd3 = ZmeasurementCircuitv2Mid(Cd3,listTarZ4LowerRight(i),listZ4(i),2);
end

for i = 1:length(listFirstCol(1,:))
    Cd3 = ZmeasurementCircuitv2LeftRight(Cd3,listFirstCol(i),listZ2cycle1(i),2);
end




% Fourth cycle (measure lower left qubits)

listTarX4LowerLeft = zeros(1,length(listX4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1) && (mod(j,2) == 0)
            listTarX4LowerLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && ((mod(j,2) == 1) && j ~= d)
            listTarX4LowerLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listHighestRowLowerLeft = listTarX4LowerLeft(1,1:length(listX2cycle1(1,:)));

listTarX4LowerLeft = listTarX4LowerLeft(1,length(listHighestRowLowerLeft(1,:))+1:length(listTarX4LowerLeft(1,:)));


% Inserts gates for measuring X of the lower right target qubits
for i = 1:length(listX4(1,:))
    Cd3 = XmeasurementCircuitv2Mid(Cd3,listTarX4LowerLeft(i),listX4(i),4);
end

for i = 1:length(listLowestRowUpperLeft(1,:))
    Cd3 = XmeasurementCircuitv2UpLow(Cd3,listHighestRowLowerLeft(i),listX2cycle1(i),4);
end

% Generate list of target qubits for listZ4
listTarZ4LowerLeft = zeros(1,length(listZ4(1,:)));

counter = 1;
for i = 1:d
    for j = 1:d
        if (mod(i,2) == 1 && (i > 1)) && ((mod(j,2) == 1) && j ~= d)
            listTarZ4LowerLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        elseif (mod(i,2) == 0) && (mod(j,2) == 0)
            listTarZ4LowerLeft(1,counter) = dataMat(i,j); 
            counter = counter + 1;
        end
    end
end

listFirstCol = zeros(1,length(listZ2cycle1(1,:)));
counter = 1;
for i = 1:d
    if (mod(i,2) == 1) && (i > 1)
        listFirstCol(1,counter) = dataMat(i,d);
        counter = counter + 1;
    end
end

% Inserts gates for measuring Z of the upper left target qubits
for i = 1:length(listZ4(1,:))
    Cd3 = ZmeasurementCircuitv2Mid(Cd3,listTarZ4LowerLeft(i),listZ4(i),4);
end

for i = 1:length(listFirstCol(1,:))
    Cd3 = ZmeasurementCircuitv2LeftRight(Cd3,listFirstCol(i),listZ2cycle2(i),4);
end




Output = Cd3;

end

function Output = ZmeasurementCircuitv2Mid(C,tarqub,datain,t0)

% t0 takes values 1,2,3 or 4 depending on the cycle (clockwise). For example, when
% measuring the upper left qubit, t0 =1, the upper right, t0 = 2, lower
% right, t0 =3 and lower left, t0 = 4. 


switch t0
    case 1   
        C(datain,4) = 1000;
        C(tarqub,4) = 1000 + datain;
    case 2       
        C(datain,7) = 1000;
        C(tarqub,7) = 1000 + datain;
    case 3        
        C(datain,10) = 1000;
        C(tarqub,10) = 1000 + datain;
    otherwise             
        C(datain,13) = 1000;
        C(tarqub,13) = 1000 + datain;
end

Output = C;

end

function Output = ZmeasurementCircuitv2LeftRight(C,tarqub,datain,t0)

% t0 takes values 1,2,3 or 4 depending on the cycle (clockwise). For example, when
% measuring the upper left qubit, t0 =1, the upper right, t0 = 2, lower
% right, t0 =3 and lower left, t0 = 4. 


switch t0
    case 1       
        C(datain,4) = 1000;
        C(tarqub,4) = 1000 + datain;
    case 2              
        C(datain,7) = 1000;
        C(tarqub,7) = 1000 + datain;
    case 3        
        C(datain,10) = 1000;
        C(tarqub,10) = 1000 + datain;
    otherwise
        C(datain,13) = 1000;
        C(tarqub,13) = 1000 + datain;
end

Output = C;

end

function Output = XmeasurementCircuitv2Mid(C,tarqub,datain,t0)

% t0 takes values 1,2,3 or 4 depending on the cycle (clockwise). For example, when
% measuring the upper left qubit, t0 =1, the upper right, t0 = 2, lower
% right, t0 =3 and lower left, t0 = 4.


switch t0
    case 1
        C(datain ,4) = 1000 + tarqub;
        C(tarqub,4) = 1000;
    case 2                
        C(datain,7) = 1000 + tarqub;
        C(tarqub,7) = 1000;
    case 3        
        C(datain,10) = 1000 + tarqub;
        C(tarqub,10) = 1000;
    otherwise        
        C(datain,13) = 1000 + tarqub;
        C(tarqub,13) = 1000;
end

Output = C;

end

function Output = XmeasurementCircuitv2UpLow(C,tarqub,datain,t0)

% t0 takes values 1,2,3 or 4 depending on the cycle (clockwise). For example, when
% measuring the upper left qubit, t0 =1, the upper right, t0 = 2, lower
% right, t0 =3 and lower left, t0 = 4.


switch t0
    case 1
                
        C(datain,4) = 1000 + tarqub;
        C(tarqub,4) = 1000;
    case 2
                
        C(datain,7) = 1000 + tarqub;
        C(tarqub,7) = 1000;
    case 3
                
        C(datain,10) = 1000 + tarqub;
        C(tarqub,10) = 1000;
    otherwise
                
        C(datain,13) = 1000 + tarqub;
        C(tarqub,13) = 1000;
end

Output = C;

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

e = zeros(1,4);
counter = 1;

for i = 1:length(Cmat(:,1))
    for j = 1:length(Cmat(1,:))
        
        % Adds storage errors with probability p/10
        if (Cmat(i,j) == 1) 
            xi = rand;
            if xi < errRate
                k = randi([1,3]);
                e(counter,:) = [k,i,1,j];
                counter = counter + 1;
            end
        end
        
        % Adds state-preparation errors with probability p
        if Cmat(i,j) == 4
            xi = rand;
            if xi < errRate
                e(counter,:) = [1,i,1,j];
                counter = counter + 1;
            end
        end
        
        % Adds Z-measurement errors with probability p
        if Cmat(i,j) == 6
            xi = rand;
            if xi < errRate
                e(counter,:) = [1,i,1,j];
                counter = counter + 1;
            end
        end
        
        % Adds Hadamard errors with probability p/4
%         if Cmat(i,j) == 10
%             xi = rand;
%             if xi < errRate
%                 k = randi([1,3]);
%                 e(counter,:) = [k,i,1,j];
%                 counter = counter + 1;
%             end
%         end
        
         % Adds CNOT errors with probability p
        if (Cmat(i,j) > 1000) && (Cmat(i,j) < 2000)
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

function [errRateX,errRateZ,errRateY,ToterrRate] = simulationSurfaceCode(errRate,d,noiseSteps,errorModel,numIterations)

n=1;

% This function will return the logical error rates of X, Y, and Z for gate error noise.

% This function takes errorModel as an input. The onvention is the following:
% 1: ErrorGeneratorMemoryOnly(Cd3Matteo,errRate)
% 2: ErrorGeneratorNoLeakage(Cd3Matteo,errRate)
% 3: ErrorGeneratorLeakageConsidered(Cd3Matteo,errRate)

if sum(errorModel == [1,2,3]) ~= 1
    disp('Invalid error model!')
    return
end

%d = 5;
%noiseSteps = 3;
%errRate = 5*10^(-3);
QerrRate = 1;
MerrRate = 1;
Cd3Matteo = circuitSurfaced3Matteov2(d);

%tic
xfaults = 0;
zfaults = 0;
yfaults = 0;
totfaults = 0;
%numIterations = 10000;

for k = 1:numIterations

TempZ = zeros(d-1,(d-1)/2+1);
TempX = zeros((d-1)/2+1,d-1);
FlagZ = [];
FlagX = [];
PastE = [];
TrackX = zeros(noiseSteps,d^2);
TrackZ = zeros(noiseSteps,d^2);
%e(1,:) = [2,70,1,1] % DEBUGGING




for t = 1:(noiseSteps+2)
    if t > noiseSteps
        e = [];
    else
        if errorModel == 1
            e = ErrorGeneratorMemoryOnly(Cd3Matteo,errRate);
        elseif errorModel == 2
            e = ErrorGenerator(Cd3Matteo,errRate);
        else
            e = ErrorGeneratorLeakageConsidered(Cd3Matteo,errRate);
        end
    end
    %%% Debugging purposes
%    AddedE = zeros(1,d^2);
%    if t == 1
%        AddedE(1,22) = 1;
%        e = errorMatConverter(AddedE,zeros(1,d^2),d);
%    end
%    if t == 2
%        AddedE(1,14) = 1;
%        e = errorMatConverter(AddedE,zeros(1,d^2),d);
%    end
    
    
    e = [e;PastE];    
    temp = PropagationStatePrepArb(Cd3Matteo, n, e);
    zErr = vecZ(temp,d);
    xErr = vecX(temp,d);
    MeasZ = MeasureZmatMatteo(temp,d);
    MeasX = MeasureXmatMatteo(temp,d);
    TempZ = mod(MeasZ+TempZ,2);
    TempX = mod(MeasX+TempX,2);
    TimeFlagZ = convertSyndromeZ(TempZ,d,t);  
    TimeFlagX = convertSyndromeX(TempX,d,t);
    FlagZ = [FlagZ;TimeFlagZ];
    FlagX = [FlagX;TimeFlagX];
    % Reset the Temp to the measured syndromes from this round
    TempZ = MeasZ;
    TempX = MeasX;
    PastE = errorMatConverter(xErr,zErr,d);
    TrackX(t,:) = xErr;
    TrackZ(t,:) = zErr;
    %e = []; %DEBUGGING
end

% FlagX;
% FlagZ;

CorrZ = minWeightMatchingX(FlagX,d,QerrRate, MerrRate);
CorrX = minWeightMatchingZ(FlagZ,d,QerrRate, MerrRate);

% CorrZsquare = reshape(CorrZ,[d,d])';
% CorrXsquare = reshape(CorrX,[d,d])';

finalZ = mod(CorrZ + zErr,2);
finalX = mod(CorrX + xErr,2);

finalZsquare = reshape(finalZ,[d,d])';
finalXsquare = reshape(finalX,[d,d])';


% e = errorMatConverter(finalX,finalZ,d); % For debugging
% temp = PropagationStatePrepArb(Cd3Matteo, n, e); % For debugging

% Check if there is a logical fault
logicalFaultX = mod(sum(finalXsquare(1,:)),2);
logicalFaultZ = mod(sum(finalZsquare(:,1)),2);

% if logicalFaultX
%     TrackX
% end
% 
% if logicalFaultZ
%     TrackZ
% end

if (logicalFaultX && logicalFaultZ)    
    yfaults = yfaults + 1;
else
    xfaults = xfaults + logicalFaultX ;
    zfaults = zfaults + logicalFaultZ;
end

if (logicalFaultX==1) || (logicalFaultZ==1) 
    totfaults = totfaults + 1;
end

%MeasZ = MeasureZmatMatteo(temp,d);
%MeasX = MeasureXmatMatteo(temp,d);

%if sum(sum(MeasX)) > 0 || sum(sum(MeasZ)) > 0
%    k
%end

end
%toc

errRateX = xfaults/numIterations;
errRateZ = zfaults/numIterations;
errRateY = yfaults/numIterations;
ToterrRate = totfaults/numIterations;

end

function [MeasXMatOut,MeasZMatOut,XErrMatOut,ZErrMatOut] = MxMzGrid(errRate,d,noiseSteps,errorModel)

n=1;

% This function will return the logical error rates of X, Y, and Z for gate error noise.

% This function takes errorModel as an input. The onvention is the following:
% 1: ErrorGeneratorMemoryOnly(Cd3Matteo,errRate)
% 2: ErrorGeneratorNoLeakage(Cd3Matteo,errRate)
% 3: ErrorGeneratorLeakageConsidered(Cd3Matteo,errRate)

if sum(errorModel == [1,2,3]) ~= 1
    disp('Invalid error model!')
    return
end

%d = 5;
%noiseSteps = 3;
%errRate = 5*10^(-3);
CMatteo = circuitSurfaced3Matteov2(d);


PastE = [];


AXOut = zeros((noiseSteps)*(d+1)/2,d-1);
AZOut = zeros((noiseSteps)*(d-1),(d+1)/2);
XerrOut = zeros(noiseSteps,d^2);
ZerrOut = zeros(noiseSteps,d^2);

for t = 1:noiseSteps
    if t > noiseSteps
        e = [];
    else
        if errorModel == 1
            e = ErrorGeneratorMemoryOnly(CMatteo,errRate);
        elseif errorModel == 2
            e = ErrorGenerator(CMatteo,errRate);
        else
            e = ErrorGeneratorLeakageConsidered(CMatteo,errRate);
        end
    end
    %%% Debugging purposes
%    AddedE = zeros(1,d^2);
%    if t == 1
%        AddedE(1,22) = 1;
%        e = errorMatConverter(AddedE,zeros(1,d^2),d);
%    end
%    if t == 2
%        AddedE(1,14) = 1;
%        e = errorMatConverter(AddedE,zeros(1,d^2),d);
%    end
    
    
    e = [e;PastE];    
    temp = PropagationStatePrepArb(CMatteo, n, e);
    zErr = vecZ(temp,d);
    xErr = vecX(temp,d);
    MeasZ = MeasureZmatMatteo(temp,d);
    MeasX = MeasureXmatMatteo(temp,d);
    
    AXOut((t-1)*((d+1)/2)+1:t*((d+1)/2),:) = MeasX;
    AZOut((t-1)*(d-1)+1:t*(d-1),:) = MeasZ;
    XerrOut(t,:) = xErr;
    ZerrOut(t,:) = zErr;
    
    PastE = errorMatConverter(xErr,zErr,d);
       
end



%toc

MeasXMatOut = AXOut; 
MeasZMatOut = AZOut;
XErrMatOut = XerrOut;
ZErrMatOut = ZerrOut;

end