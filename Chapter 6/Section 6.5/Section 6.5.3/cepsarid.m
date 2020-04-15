function alpha = cepsarid(c,p)
   alpha = ones(p+1,1);
   for k = 1:p
       B = 0;
       for l = 0:k-1
            B = B -factorial(l+1)*comb(k-1,l)*alpha(k-l)*c(l+2);
       end
       alpha(k+1) = B;
   end
   
   for k = 1:p
       alpha(k+1) = alpha(k+1)/factorial(k);
   end
end

function comb = comb(n,i)
    comb = factorial(n)/(factorial(i)*factorial(n-i));
end