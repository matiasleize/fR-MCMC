## oregonator differential equation

function xdot = f(x, z)
  
  xdot = zeros (4,1);
 
  #x = x(1);
  %y = x(2);
  %v = x(3);
  #w = x(4);
  
  gama= @(A,B) 0.5 .* (A .* B) ./((B-A).^2);
  %gama= @(A,B) A./n*(A.-B);

  xdot(1) = (-x(4) + x(1)**2 + (1 + x(3))*x(1) - 2*x(3) + 4*x(2))/(1+z);
  
  xdot(2) = -(x(3)*x(1)*gama(x(2), x(3)) - x(1)*x(2)  + 4*x(2) - 2*x(2)*x(3))/(1+z);
  
  xdot(3) = -(x(3) * (x(1)*gama(x(2), x(3))  + 4 - 2*x(3)))/(1+z);
  
  xdot(4) = (x(4) * (-1 + x(1) + 2*x(3)))/(1+z);
  
endfunction
