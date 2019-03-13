xm=-pi/4:pi/16:7*pi/4;
y1=10*sqrt(2)*sin(xm+pi/4);
y2=15*sqrt(2)*sin(xm+pi/4);
y3=30*sqrt(2)*sin(xm+pi/4);
plot(xm,y1,xm,y2,xm,y3)
legend('y1=10*sqrt(2)*sin(x+pi/4)','y2=15*sqrt(2)*sin(x+pi/4)','y3=30*sqrt(2)*sin(x+pi/4)')
syms x y
eqns =[10*sqrt(2)*sin(x+pi/4)==y,15*sqrt(2)*sin(x+pi/4)==y];
S=solve(eqns,[x,y]);
X=double(S.x(2));
Y=double(S.y(2));
str2 = ['x=',num2str(X),' y=',num2str(Y)];
text(X,Y,str2);
grid