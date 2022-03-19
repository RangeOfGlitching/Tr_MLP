clear;
load out.dat
load act_y.dat
plot(out, 'o');
hold;
plot(act_y,'x');
err= out - act_y;
dim= size(out);
(sum(err.^2)/dim(1,1))^0.5


