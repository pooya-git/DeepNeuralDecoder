function output = convertSyndromeX(flagM,dist,time)% This function takes as input the matrix representing the flagged syndromes% and converts it to a list of coordinated in space-time for the decodernumflags = sum(sum(flagM));if numflags > 0    output = zeros(numflags,3);    counter = 1;    for i = 1:((dist-1)/2+1)        for j = 1:(dist-1)            if flagM(i,j) == 1                output(counter,:) = [j+1,2*i-mod(j+1,2),time];                counter = counter + 1;            end                end    endelse    output = [];    endend