x =  -10:0.5:10;
y = 1./(1+exp(-x));

plot(x,y)
title('Logistic Regression for Aclass')
ylabel('Probability for a sample that belongs to Aclass');
xlabel('w^T * x');
grid on;
