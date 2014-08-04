addpath(genpath('C:/users/GVIS Demo/calvin/airline-allocation'))
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/inputs_before_3routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/outputs_before_3routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/coefficients_before_3routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/constants_before_3routes.mat'

disp("3routres")
OverrideFunction_3routes(Inputs, Outputs, Coefficients, Constants);

load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/inputs_before_11routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/outputs_before_11routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/coefficients_before_11routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/constants_before_11routes.mat'

disp("11routres")
OverrideFunction_11routes(Inputs, Outputs, Coefficients, Constants);

load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/inputs_before_31routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/outputs_before_31routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/coefficients_before_31routes.mat'
load 'C:/Users/GVIS Demo/calvin/airline-allocation/Data/constants_before_31routes.mat'

disp("31routres")
OverrideFunction_31routes(Inputs, Outputs, Coefficients, Constants);
