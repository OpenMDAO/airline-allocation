% Override Function
function [Inputs,Outputs,Constants,Coefficients] = OverrideFunction_31routes(Inputs,Outputs,Coefficients,Constants)

       ac_ind = [6;10;4;9;3;8];
    Inputs.ACNum = [1;7;2;8;19;1];
    distance = [113;174;289;303;324;331;342;375;407;427;484;486;531;543;550;570;594;609;622;...
        680;747;758;760;823;837;991;1098;1231;1407;1570;1626];
    Inputs.DVector = [[1:31]' , [99;80;51;184;263;169;158;135;284;184;92;132;...
        754;150;238;264;365;749;234;50;101;124;175;221;258;105;112;129;506;134;266]];
    

    %%
    route_ind = range_extract(Inputs,distance);
    Inputs.RVector = Inputs.RVector(route_ind);
    Inputs.AvailPax = Inputs.AvailPax(ac_ind); 
    Inputs.TurnAround = 1; 
    J = length(Inputs.DVector(:,2));   %Number of routes              
    K = length(Inputs.AvailPax);        % Number of Aircraft types
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
                +Inputs.TurnAround)); % +1 to account for zero # of trip
            rw = rw + 1;
        end
    end
    Inputs.MaxTrip = max_trip;