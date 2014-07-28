% Extract range indices
function [ind] = range_extract(Inputs,distance)

ind = zeros(length(distance),1);

for cc  = 1:length(distance)
    range = distance(cc);
    diff_min = inf;
    for ii = 1:length(Inputs.RVector)
            diff = abs(Inputs.RVector(ii) - range);
            if diff<diff_min
                  ind(cc) = ii;
                  diff_min = diff;
            end
    end
end
        
        
        
        
        


