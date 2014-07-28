% Test problem
% clc;close all;clear all
% x = [9/4;15/4;1200;500];
% A = [1,1,3,3;4,1,6,5;1,1,0,0;5,9,0,0]
% b = [12;1;6;45]
% Aeq=[];
% beq =[];
% ind_con = 1:2;
% ind_int = 3:4;
% indeq_con = [];
% indeq_int = [];
% num_int = 2;

% Call the cutting plane algorithm
% Extracts out only the integer design variables and their associated
% constrain matrices
% Important: Assumes the design vector as x = [x_integer;x_continuous]
function [ A_up, b_up] = call_Cutplane(x,A,b,Aeq,beq,ind_con,ind_int,indeq_con,indeq_int,num_int)
num_con = length(x) - num_int;
x_trip = x(1:num_int);
pax = x(num_int+1:end);

if isempty(b)~=1
    % A can subdivided into 4 matrices
    % A = [A_x_con,A_pax_con;
    %           A_x_int,A_pax_int]
    A_x_int = A(ind_int,1:num_int);
    A_pax_int = A(ind_int,num_int+1:end);
    b_x_int = b(ind_int) - A_pax_int*pax;
else
    A_x_int = [];
    b_x_int = [];
end

if isempty(beq)~=1
    Aeq_x_int = Aeq(indeq_int,1:num_int);
    Aeq_pax_int = Aeq(indeq_int,num_int+1:end);
    beq_x_int = beq(indeq_int) - Aeq_pax_int*pax;
else
    Aeq_x_int = [];
    beq_x_int = [];
end

[A_x_int_up, b_x_int_up,eflag] = GomoryCut(x_trip,A_x_int,b_x_int, Aeq_x_int,beq_x_int);

if eflag == 1
    A_new = [A_x_int_up(end,:),ones(1,num_con)];
    b_new = b_x_int_up(end) + ones(1,num_con)*pax;
else
    A_new =[];
    b_new = [];
end
A_up = [A;A_new];
b_up = [b;b_new];

