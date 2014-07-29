%% Folder Mapping v1.0
% 15 October, 2011

% This file maps all the sub-folders within the main folder for FLEET.

d=genpath(pwd);
[t,s]=strtok(d,pathsep());
while(~isempty(s))
   if(isempty(regexp(t,'(.svn|.git)')))
    addpath(t);
   end
   [t,s]=strtok(s,pathsep());
end
clear d
clear t
clear s