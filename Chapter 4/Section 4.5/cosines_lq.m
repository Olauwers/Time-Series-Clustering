function cosines = cosines_lq(A,B)

p = size(A,1);
q = size(B,1);
L = triu(qr([A;B]'));
L = L(1:p+q,p+1:p+q);
S = triu(qr(L));
S = S(1:q,:);
L = L(1:p,:);
cosines = svd(L/S);
