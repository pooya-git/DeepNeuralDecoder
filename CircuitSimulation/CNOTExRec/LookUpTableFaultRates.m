function test

    p = [1.0*10^-3 
         1.0*10^-4
         1.5*10^-3
         2.0*10^-3
         2.0*10^-4
         2.5*10^-3
         3.0*10^-3
         3.0*10^-4
         4.0*10^-4
         5.0*10^-4
         6.0*10^-4
         7.0*10^-4
         8.0*10^-4
         8.0*10^-5
         8.5*10^-5
         9.0*10^-4
         9.0*10^-5
         9.5*10^-5]

    disp('8.000*10^-5')
    res = zeros(length(p), 3);

    for err_num=1:length(p)
        load(strcat('errVecFull', sprintf('%.3d', p(err_num)), '.mat'));
        err= 0;
        sigma= 0;

        for i=1:15
            temp_err = errorVecMat(1,i)/(10^7);
            err = err + temp_err;
            sigma= sigma + sqrt(temp_err*(1-temp_err)/(10^7));
        end
        res(err_num, 1)= p(err_num);
        res(err_num, 2)= err;
        res(err_num, 3)= sigma;
    end

    res
    
end
