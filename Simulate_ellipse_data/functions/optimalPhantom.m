function [data] = optimalPhantom(object, M, D, E)

Eeff = calculateEffectivekVAll(E, D);
muEff = zeros([length(Eeff), size(M,2)]);
for ii = 1:length(Eeff)
    [c I] = min(abs(E-Eeff(ii)));
    muEff(ii,:) = M(I,:);
end

data = zeros(size(object, 1), size(D, 1));
for ii = 1:size(object, 2)
    
    objectIter = object(:, ii) > 0;
    
    data = data + muEff(:,ii)'.*objectIter;
    
end


 
end