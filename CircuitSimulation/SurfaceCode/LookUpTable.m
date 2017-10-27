GX = [0,1,1,0,0,0,0,0,0;
      1,1,0,1,1,0,0,0,0;
      0,0,0,0,0,0,1,1,0;
      0,0,0,0,1,1,0,1,1];
  
XL = [0,0,1,0,1,0,1,0,0];

% Store syndrome bit strings into vector s
s = zeros(1,4);

% generate all 15 bit strings of weight 1,2 and 3 in matrix A
A = zeros(15,9);
row = 1;

% generate rows of bit strings of weight 1
for i = 1:9
    A(row,i) = 1;
    row = row + 1;
end
  
% generate rows of bit strings of weight 2
for i = 1:9
    for j = i:9
        if i ~= j
            A(row,i) = 1;
            A(row,j) = 1;
            row = row + 1;
        end
    end    
end

% % generate rows of bit strings of weight 3
% for i = 1:9
%     for j = i:9
%         for k = j : 9
%             if (i ~= j) && (i ~= k) && (j ~= k)
%                 A(row,i) = 1;
%                 A(row,j) = 1;
%                 A(row,k) = 1;
%                 row = row + 1;
%             end
%         end
%     end
% end

% Generate a matrix where each row i represents the syndrome corresponding to
% the row i of A
Syn = zeros(length(A),4);
for i = 1:length(A)
    for j = 1:4
        Syn(i,j) = mod(sum(conj(A(i,:)).* GX(j,:)),2);
    end
end

C = [Syn,A];
[Cnew,ia,ic] = unique(C(:,1:4),'rows');
newMat = zeros(length(ia),13);
for i = 1:length(ia)
    newMat(i,:) = C(ia(i),:);
end

newMatX = zeros(16,9);
newMatX(2:16,:) = newMat(2:16,5:13);
newMatX



% -------------------Look up table for Z-stabilizers----------------------

GZ = [1,0,0,1,0,0,0,0,0;
      0,1,1,0,1,1,0,0,0;
      0,0,0,1,1,0,1,1,0;
      0,0,0,0,0,1,0,0,1];
  
ZL = [0,0,1,0,1,0,1,0,0];

% Store syndrome bit strings into vector s
s = zeros(1,4);

% generate all 15 bit strings of weight 1,2 and 3 in matrix A
A = zeros(15,9);
row = 1;

% generate rows of bit strings of weight 1
for i = 1:9
    A(row,i) = 1;
    row = row + 1;
end
  
% generate rows of bit strings of weight 2
for i = 1:9
    for j = i:9
        if i ~= j
            A(row,i) = 1;
            A(row,j) = 1;
            row = row + 1;
        end
    end    
end

% % generate rows of bit strings of weight 3
% for i = 1:9
%     for j = i:9
%         for k = j : 9
%             if (i ~= j) && (i ~= k) && (j ~= k)
%                 A(row,i) = 1;
%                 A(row,j) = 1;
%                 A(row,k) = 1;
%                 row = row + 1;
%             end
%         end
%     end
% end

% Generate a matrix where each row i represents the syndrome corresponding to
% the row i of A
Syn = zeros(length(A),4);
for i = 1:length(A)
    for j = 1:4
        Syn(i,j) = mod(sum(conj(A(i,:)).* GZ(j,:)),2);
    end
end

C = [Syn,A];
[Cnew,ia,ic] = unique(C(:,1:4),'rows');
newMat = zeros(length(ia),13);
for i = 1:length(ia)
    newMat(i,:) = C(ia(i),:);
end

newMatZ = zeros(16,9);
newMatZ(2:16,:) = newMat(2:16,5:13);
newMatZ