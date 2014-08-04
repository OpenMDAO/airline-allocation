%% GamsOutputsGen

% Generating Outputss from GAMS allocation solution
% 10-11-2012

function [Outputs,Inputs] = OutputGen_AllCon(xopt,fopt,Inputs,Constants,Coefficients,Outputs)

J = length(Inputs.DVector(:,2));   %Number of routes              
K = length(Inputs.AvailPax);        % Number of Aircraft types

x_hat = xopt(1:K*J); %Airline allocation variable
aa = find(abs(x_hat-0)<1e-6);
x_hat(aa) = 0;
pax = xopt(K*J+1:K*J*2); %passenger design variable
bb = find(abs(pax-0)<1e-6);
pax(bb) = 0;

NoiseData = Inputs.NoiseData;
DVector = Inputs.DVector;
RVector = Inputs.RVector;
T  = Inputs.T;
kk = Inputs.kk;
I = length(DVector);     % route index
% J = length(Constants.Runway); % airport index
% K = length(Inputs.AvailPax);   % aircraft class index

detailtrips = zeros(K,J);
pax_rep = zeros(K,J);
ind_pax = 0;
for k = 1:K
    for j = 1:J
        ind = (k-1)*J + j;
        detailtrips(k,j) = 2*x_hat(ind);
        pax_rep(k,j) = 2*pax(ind);
    end
end
         
[r,c] = size(detailtrips);
Outputs.DetailTrips = detailtrips;
unit_detailtrips=(detailtrips./detailtrips);
bb=find(detailtrips==0);
unit_detailtrips(bb)=0;

for i = 1:r
    aa(i,1) = length(find(detailtrips(i,:)~=0));
    Outputs.Trips(i,1)  = sum(detailtrips(i,:));
     Outputs.FleetUsed(i,1)  = ceil(sum(Coefficients.BlockTime(i,:).*(1+Constants.MH(i,1)).*detailtrips(i,:) + detailtrips(i,:).*Inputs.TurnAround)./24);
%     Outputs.FleetUsedAll(i,:)  = ((Coefficients.BlockTime(i,:).*(1+Constants.MH(i,1)).*detailtrips(i,:) + detailtrips(i,:).*Inputs.TurnAround)./24);
    Outputs.Fuel(i,1)   = sum(Coefficients.Fuelburn(i,:).*detailtrips(i,:));
%     acq_unused(i,1)    = Coefficients.Acq(i).*(Inputs.ACNum(i)-Outputs.FleetUsed(i,1));
    Outputs.Doc(i,1)    = sum(Coefficients.Doc(i,:).*detailtrips(i,:));
    Outputs.BlockTime(i,1)= sum(Coefficients.BlockTime(i,:).*detailtrips(i,:));
    Outputs.Nox(i,1)    = sum(Coefficients.Nox(i,:).*detailtrips(i,:));
    Outputs.Maxpax(i,1) = sum(Inputs.AvailPax(i,1).*detailtrips(i,:));
    Outputs.Pax(i,1)    = sum(pax_rep(i,:));
    Outputs.Miles(i,1) = sum(pax_rep(i,:).*RVector');
end

Outputs.CostDetail  = Coefficients.Doc.*detailtrips + Coefficients.Fuelburn.*Constants.FuelCost(kk).*detailtrips;
Outputs.RevDetail   = Outputs.TicketPrice.*pax_rep;
Outputs.PaxDetail   = pax_rep;
Outputs.RevArray    = sum(Outputs.RevDetail,1);
Outputs.CostArray   = sum(Outputs.CostDetail,1);
Outputs.PaxArray    = sum(pax_rep,1);
Outputs.ProfitArray = (Outputs.RevArray - Outputs.CostArray);
Outputs.Revenue      = sum(Outputs.RevDetail,2);

% record a/c performance
PPNM=zeros(1,K);
ProfitArray=Outputs.ProfitArray;
profit_v = sum(ProfitArray');

den_v= sum(Outputs.PaxArray.*RVector');
PPNM = profit_v./den_v; 
for i=1:length(PPNM)
    if isnan(PPNM(i))
        PPNM(i)=0;
    end
end

Outputs.Cost         = sum(Outputs.Doc + Outputs.Fuel.*Constants.FuelCost(kk));
Outputs.PPNM         = PPNM;
Outputs.Profit       = sum(Outputs.RevArray - Outputs.CostArray);

%% Allocation detail info
for i = i:length(RVector)
    a = find(detailtrips(:,i));
    info = [a detailtrips(a,i) pax_rep(a,i)];
    Outputs.Info(i,1) = {info};
end
