%% Constraint function
%Generates the constraint coefficient for the problem
function [Ain,bin] = AirlineAllocateCon(Inputs,Constants,Coefficients)

 J = length(Inputs.DVector(:,2));   %Number of routes              
 K = length(Inputs.AvailPax);        % Number of Aircraft types
  
%% Linear Inequality constraints (Ain*x <= bin)
dem = Inputs.DVector(:,2);
BH = Coefficients.BlockTime;
MH = Constants.MH;
cap  = Inputs.AvailPax;
fleet = Inputs.ACNum;
t = Inputs.TurnAround;

% Upper demand constraint
 Ain_1 = zeros(J,2*K*J);
 bin_1 = dem;
 for jj = 1:J
      for kk = 1:K
          col = K*J + (kk-1)*J + jj;
          Ain_1(jj,col) = 1;
      end
 end
 
 % Lower demand constraint
 Ain_2 = zeros(J,2*K*J);
 bin_2 = -0.2*dem;
  for jj = 1:J
      for kk = 1:K
           col = K*J + (kk-1)*J + jj;
           Ain_2(jj,col) = -1;
      end
  end
 
 %Aircraft utilization constraint
 Ain_3 = zeros(K,K*J*2);
 bin_3 = zeros(K,1);
 for kk = 1:K
     for jj = 1:J 
        col = (kk-1)*J + jj;
        Ain_3(kk,col) = BH(kk,jj)*(1+MH(kk,1))+t;
     end
     bin_3(kk,1) = 12*fleet(kk);
 end
 
 %Aircraft capacity constraint
 Ain_4 = zeros(K*J,K*J*2);
 bin_4 = zeros(K*J,1);
 rw = 1;
 for kk = 1:K
     for jj = 1:J    
         col1 = (kk-1)*J + jj;
         Ain_4(rw,col1) = -cap(kk);
         col2 = K*J + (kk-1)*J + jj;
         Ain_4(rw,col2) = 1;
         rw = rw + 1;
     end
 end
  
Ain = [Ain_1;Ain_2;Ain_3;Ain_4];
bin = [bin_1;bin_2;bin_3;bin_4];

