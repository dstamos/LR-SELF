function dagA = fasOpt(A, exact)

    if nargin < 2
        exact = 0;
    end
    
    %A = (A - A')/2;
    %A = A.*(A>=0);
    
    nvert = size(A,1);
    
    ind = (1:nvert)'*ones(1,nvert);
    tind = ind';
    pos = [tind(:), ind(:)];
    tA = A';
    weights = tA(:);
    dz = weights ~= 0;
    weights = weights(dz);
    pos = pos(dz,:);
    tpos = pos';
    edges = tpos(:);
    

    r = mexino(edges-min(edges), weights, exact);
    for i = r'
        A(pos(i+1,1),pos(i+1,2)) = 0;
    end
    dagA = A;
end