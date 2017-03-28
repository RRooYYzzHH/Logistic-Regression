function [ w, e_in, iter] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)
    e_in = 0;
    w = w_init;
    [row, col] = size(X);
    for iter = 1:1:max_its
        %xnew = [ones(row, 1), X];
        %t = xnew * w';
        %grad = y' .* xnew ./ (1 + exp(y' * t));
        grad = zeros(1, 14);
        for n = 1:1:row       
            xnew = [1, X(n, :)];
            t = w * xnew';
            grad = grad + y(n, 1) * xnew / (1 + exp(y(n, 1) * t));
        end
        grad = grad / row;
        
        judge = 1;
        for i = 1:1:(col + 1)
            if abs(grad(1, i)) >= 0.000001
                judge = 0;
            end
        end
        if judge == 1
            break;
        end
        w = w + eta * grad;
    end
    
    fprintf('%d', iter)
    
    for n = 1:1:row
        xnew = [1, X(n, :)];
        t = w * xnew';
        a = - y(n, 1) * t;
        if a >= 600
            e_in = e_in + a;
        else
            e_in = e_in + log(1 + exp(- y(n, 1) * t));
        end  
    end
    e_in = e_in / row;
end

