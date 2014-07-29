%% Objective function for MILP -> Continuous problem 
% Generates the objecitve coeffcients of the problem
function [obj_int,obj_con] = AirlineAllocateObj(Inputs,Outputs,Constants,Coefficients)

 J = length(Inputs.DVector(:,2));                 
 K = length(Inputs.AvailPax);        
 
 fuelburn = Coefficients.Fuelburn;
 docnofuel = Coefficients.Doc;
 Price = Outputs.TicketPrice;
 fuelcost = Constants.FuelCost;
 
 obj_int = zeros(K*J,1);
 obj_con = zeros(K*J,1);
 
 for kk = 1:K
     for jj = 1:J
         col = (kk-1)*J + jj;
         obj_int(col) = docnofuel(kk,jj) + fuelcost*fuelburn(kk,jj);
         obj_con(col) = -Price(kk,jj);
     end
 end
 