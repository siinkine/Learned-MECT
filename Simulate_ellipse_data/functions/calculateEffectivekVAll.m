function Eeff = calculateEffectivekVAll(E, S)
% calculate effective kV
Eeff = [];
for ii = 1:size(S, 1)
    
    Eeff(ii) = sum(S(ii, :).*E') / (sum(S(ii, :)));    
    
end

end