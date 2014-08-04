%% This is the main file
% This loads the large network dataset (2134 routes and 18 types of
% aricraft)
% Creates a new smaller network using the override function
% Calls the branch and cut algorithm to solve the MILP problem

clear all
clc
close all
% Map Folders
SetPath
cd Output_3routes

% Load the Dataset 
% Provides performance & cost coefficents  of aircraft
% Demand and ticket price details of the airline routes
load Dataset
%% Override the required parameters for the new network 
[Inputs,Outputs,Constants,Coefficients] = OverrideFunction_3routes(Inputs,Outputs,Coefficients,Constants);

% calculate the objective and constraint coefficient matrix/vector
[Ain,bin] = AirlineAllocateCon(Inputs,Constants,Coefficients);
[obj_int,obj_con] = AirlineAllocateObj(Inputs,Outputs,Constants,Coefficients);

%Initialize 
J = length(Inputs.DVector(:,2));   %Number of routes              
K = length(Inputs.AvailPax);        % Number of Aircraft types
lb = zeros(2*K*J,1);
ub = [Inputs.MaxTrip.*ones(K*J,1);Inf*ones(K*J,1)]; 
x0 = [];
ind_conCon = 1:2*J;
ind_intCon = 2*J+1:size(Ain,1);

tic  %Start the timer for the branch and cut algorithm
[xopt,fopt,can_x,can_F,x_best_relax,f_best_relax,funCall,exitflag] = branch_cut...
    (obj_int,obj_con,Ain,bin,[],[],lb,ub,x0,ind_conCon,ind_intCon,[],[]);
runtime = toc;
save('runtime.mat','runtime')

if exitflag == 1
    [Outputs,Inputs] = OutputGen_AllCon(xopt,fopt,Inputs,Constants,Coefficients,Outputs);
    NetProfit = Outputs.Profit;DetailTrips = Outputs.DetailTrips;DetailPax = Outputs.PaxDetail;
    save('Result.mat','NetProfit','DetailTrips','DetailPax','funCall')
end
cd ..
