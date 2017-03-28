M = csvread('clevelandtrain.csv', 1, 0);
[row, col] = size(M);
for n = 1:1:152
   if M(n,14) == 0
        M(n, 14) = -1;
   end
end
X = M(:,1:13);
y = M(:,14);
w_init = zeros(1, 14);
[w1, ein1] = logistic_reg(X, y, w_init, 10000000, 0.00001);


