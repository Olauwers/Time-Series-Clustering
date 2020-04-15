function R = projected(A,B)

p = size(A,1);
q = size(B,1);
L = triu(qr([A;B].')).';
L22 = L(p+1:p+q,p+1:p+q);
[U,S,V] = svd(L22);
R= U(:,1:rank(S,sqrt(eps))).';
