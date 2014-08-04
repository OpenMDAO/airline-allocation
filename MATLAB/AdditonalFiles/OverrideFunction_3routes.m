% Override Function
% this function extracts the sub network data from the dataset 
function [Inputs,Outputs,Constants,Coefficients] = OverrideFunction_3routes(Inputs,Outputs,Coefficients,Constants)

  %% Custom Inputs
    ac_ind = [9;10];     %index for aircraft
    distance = [2000;1500;1000]; %Route distance
    Inputs.DVector = [[1:3]',[300;700;220]]; %Route demand
    Inputs.ACNum = [6;4]; %Number of aircraft 
%%
    route_ind = range_extract(Inputs,distance)
    Inputs.RVector = Inputs.RVector(route_ind);
    Inputs.AvailPax = Inputs.AvailPax(ac_ind); 
    Inputs.TurnAround = 1; 
    J = length(Inputs.DVector(:,2))   %Number of routes              
    K = length(Inputs.AvailPax)       % Number of Aircraft types
    Inputs.Lim = ones(K,J);
    
    Constants.Runway = 1e4.*length(Inputs.RVector);
    Constants.MH = Constants.MH(ac_ind);
    Constants.FuelCost = 0.2431;
    Constants.demfac      = 1;
    
    TicketPrice = [];
    tempTP = Outputs.TicketPrice;
    for kk = 1:length(ac_ind)
         TP = []; 
        for jj = 1:length(route_ind)
            TP = [TP,tempTP(ac_ind(kk),route_ind(jj))];
        end
        TicketPrice = [TicketPrice;TP];
    end
    Outputs.TicketPrice = TicketPrice;
    clear tempTP TP
    
    Fuelburn = [];
    tempCoef = Coefficients.Fuelburn;
    for kk = 1:length(ac_ind)
        tc = []; 
        for jj = 1:length(route_ind)
            tc = [tc,tempCoef(ac_ind(kk),route_ind(jj))];
        end
        Fuelburn = [Fuelburn;tc];
    end
    Coefficients.Fuelburn = Fuelburn;
    clear tempCoef tc
    
    Doc = [];
    tempCoef = Coefficients.Doc;
    for kk = 1:length(ac_ind)
        tc = []; 
        for jj = 1:length(route_ind)
            tc = [tc,tempCoef(ac_ind(kk),route_ind(jj))];
        end
        Doc = [Doc;tc];
    end
    Coefficients.Doc = Doc;
    clear tempCoef tc
     
    Nox =[];
    tempCoef = Coefficients.Nox;
    for kk = 1:length(ac_ind)
        tc = []; 
        for jj = 1:length(route_ind)
            tc = [tc,tempCoef(ac_ind(kk),route_ind(jj))];
        end
        Nox = [Nox;tc];
    end
    Coefficients.Nox = Nox;
    clear tempCoef tc
     
    BlockTime = [];
    tempCoef = Coefficients.BlockTime;
    for kk = 1:length(ac_ind)
        tc = [];
        for jj = 1:length(route_ind)
            tc = [tc,tempCoef(ac_ind(kk),route_ind(jj))];
        end
        BlockTime = [BlockTime;tc];
    end
    Coefficients.BlockTime = BlockTime;
    clear tempCoef tc
    
     %Max trip (Upper bounds on design variables)
    rw = 1;
    max_trip = zeros(K*J,1);
    for kk = 1:K
        for jj = 1:J
            max_trip(rw) = Inputs.ACNum(kk)*ceil(12/(BlockTime(kk,jj)*(1+Constants.MH(kk))...
                +Inputs.TurnAround)); 
            rw = rw + 1;
        end
    end
    Inputs.MaxTrip = max_trip;