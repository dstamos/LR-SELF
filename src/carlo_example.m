A = [0, 2, 0; ...
     0, 0, 5; ...
     1, 0, 0];
 
A

exact = 1;
dagA = fasOpt(A, exact);

dagA






%%






























































%%

mex mexino.c -I'C:\Users\dstamos\Downloads\igraph-0.7.1-msvc\igraph-0.7.1-msvc\include' -L'C:\Users\dstamos\Downloads\igraph-0.7.1-msvc\igraph-0.7.1-msvc\Debug' -ligraph